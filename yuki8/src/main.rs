#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque},
};

mod rnd {
    static mut S: usize = 0;
    static MAX: usize = 1e9 as usize;

    #[inline]
    pub fn init(seed: usize) {
        unsafe {
            if seed == 0 {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as usize;
                S = t
            } else {
                S = seed;
            }
        }
    }
    #[inline]
    pub fn gen() -> usize {
        unsafe {
            if S == 0 {
                init(0);
            }
            S ^= S << 7;
            S ^= S >> 9;
            S
        }
    }
    #[inline]
    pub fn gen_range(a: usize, b: usize) -> usize {
        gen() % (b - a) + a
    }
    #[inline]
    pub fn gen_bool() -> bool {
        gen() & 1 == 1
    }
    #[inline]
    pub fn gen_range_isize(a: usize) -> isize {
        let mut x = (gen() % a) as isize;
        if gen_bool() {
            x *= -1;
        }
        x
    }
    #[inline]
    pub fn gen_range_neg_wrapping(a: usize) -> usize {
        let mut x = gen() % a;
        if gen_bool() {
            x = x.wrapping_neg();
        }
        x
    }
    #[inline]
    pub fn gen_float() -> f64 {
        ((gen() % MAX) as f64) / MAX as f64
    }
}

#[derive(Debug, Clone)]
struct TimeKeeper {
    start_time: std::time::Instant,
    time_threshold: f64,
}

impl TimeKeeper {
    fn new(time_threshold: f64) -> Self {
        TimeKeeper {
            start_time: std::time::Instant::now(),
            time_threshold,
        }
    }
    #[inline]
    fn isTimeOver(&self) -> bool {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 1.5 >= self.time_threshold
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time >= self.time_threshold
        }
    }
    #[inline]
    fn get_time(&self) -> f64 {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 1.5
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

const DIJ4: [(usize, usize); 4] = [(!0, 0), (0, !0), (1, 0), (0, 1)];

#[derive(Debug, Clone)]
struct State {
    N: usize,
    A: Vec<Vec<char>>,
    B: Vec<Vec<char>>,
    b: usize,
    cost: usize,
    building_num: usize,
    bomb_store_num: usize,
    building_destruct_cnt: usize,
    bomb_store_destruct_cnt: usize,
    explosion_positions: Vec<(usize, usize, usize)>,
    actions: Vec<(usize, usize)>,
}

impl State {
    fn new(N: usize) -> Self {
        input! {
            A: [chars; N]
        }
        let mut building_num = 0;
        let mut bomb_store_num = 0;
        for i in 0..N {
            for j in 0..N {
                if A[i][j] == '#' {
                    building_num += 1;
                } else if A[i][j] == '@' {
                    bomb_store_num += 1;
                }
            }
        }
        State {
            N,
            A: A.clone(),
            B: A,
            b: 0,
            cost: 0,
            building_num,
            bomb_store_num,
            building_destruct_cnt: 0,
            bomb_store_destruct_cnt: 0,
            explosion_positions: vec![],
            actions: vec![],
        }
    }
    fn init(&mut self, bombs: &[Bomb]) -> (usize, usize, Vec<bool>) {
        let mut store_positions = vec![];
        for y in 0..self.N {
            for x in 0..self.N {
                if self.A[y][x] == '@' {
                    store_positions.push((y, x, 0));
                }
            }
        }
        let (idx, a, c) = self.bfs(0, 0, &store_positions);
        self.actions.extend(a);
        self.cost += c;
        let (sy, sx, _) = store_positions[idx];

        let mut used = vec![false; bombs.len()];
        let mut A = self.A.clone();

        for _ in 0..10 {
            let mut candidate = vec![];
            for i in 0..bombs.len() {
                if used[i] {
                    continue;
                }
                let mut destruct_cnt = 0;
                for (a, b) in bombs[i].AB.iter() {
                    let ny = sy as isize + a;
                    let nx = sx as isize + b;
                    if ny < 0 || self.N as isize <= ny || nx < 0 || self.N as isize <= nx {
                        continue;
                    }
                    let ny = ny as usize;
                    let nx = nx as usize;
                    if self.B[ny][nx] != '.' {
                        destruct_cnt += 1;
                    }
                }
                candidate.push((destruct_cnt, i));
            }
            candidate.sort_by_key(|x| Reverse(x.0));
            let (_, idx) = candidate[0];
            let before_A = A.clone();

            for (a, b) in bombs[idx].AB.iter() {
                let ny = sy as isize + a;
                let nx = sx as isize + b;
                if ny < 0 || self.N as isize <= ny || nx < 0 || self.N as isize <= nx {
                    continue;
                }
                let ny = ny as usize;
                let nx = nx as usize;
                A[ny][nx] = '.';
            }

            let mut store_cnt = 0;
            for y in 0..self.N {
                for x in 0..self.N {
                    if A[y][x] == '@' {
                        store_cnt += 1
                    }
                }
            }
            if store_cnt == 0 {
                A = before_A;
                break;
            } else {
                used[idx] = true;
            }
        }

        let mut building_cnt = 0;
        let mut bomb_store_cnt = 0;
        for y in 0..self.N {
            for x in 0..self.N {
                if A[y][x] == '#' {
                    building_cnt += 1
                } else if A[y][x] == '@' {
                    bomb_store_cnt += 1;
                }
            }
        }
        self.building_destruct_cnt = self.building_num - building_cnt;
        self.bomb_store_destruct_cnt = self.bomb_store_num - bomb_store_cnt;
        self.A = A.clone();
        self.B = A;

        (sy, sx, used)
    }
    fn destruct_A(&mut self, y: usize, x: usize, bomb: &Bomb) {
        for (a, b) in bomb.AB.iter() {
            let ny = y as isize + a;
            let nx = x as isize + b;
            if ny < 0 || self.N as isize <= ny || nx < 0 || self.N as isize <= nx {
                continue;
            }
            let ny = ny as usize;
            let nx = nx as usize;
            self.A[ny][nx] = '.';
        }
    }
    fn check_destruct_A(&mut self, y: usize, x: usize, bomb: &Bomb) -> bool {
        let mut A = self.A.clone();
        for (a, b) in bomb.AB.iter() {
            let ny = y as isize + a;
            let nx = x as isize + b;
            if ny < 0 || self.N as isize <= ny || nx < 0 || self.N as isize <= nx {
                continue;
            }
            let ny = ny as usize;
            let nx = nx as usize;
            A[ny][nx] = '.';
        }
        let mut store_cnt = 0;
        for y in 0..self.N {
            for x in 0..self.N {
                if A[y][x] == '@' {
                    store_cnt += 1;
                }
            }
        }
        store_cnt > 0
    }
    fn destruct_B(&mut self, y: usize, x: usize, bomb: &Bomb) {
        for (a, b) in bomb.AB.iter() {
            let ny = y as isize + a;
            let nx = x as isize + b;
            if ny < 0 || self.N as isize <= ny || nx < 0 || self.N as isize <= nx {
                continue;
            }
            let ny = ny as usize;
            let nx = nx as usize;
            if self.B[ny][nx] == '#' {
                self.B[ny][nx] = '.';
                self.building_destruct_cnt += 1;
            } else if self.B[ny][nx] == '@' {
                self.B[ny][nx] = '.';
                self.bomb_store_destruct_cnt += 1;
            }
        }
    }
    fn is_destructed(&self) -> bool {
        self.building_destruct_cnt == self.building_num
            && self.bomb_store_destruct_cnt == self.bomb_store_num
    }
    fn get_explosion_positions(&mut self, bombs: &[Bomb]) {
        while !self.is_destructed() {
            let mut candidate = vec![];
            for y in 0..self.N {
                for x in 0..self.N {
                    for (i, bomb) in bombs.iter().enumerate() {
                        let mut destruct_cnt = 0;
                        for (a, b) in bomb.AB.iter() {
                            let ny = y as isize + a;
                            let nx = x as isize + b;
                            if ny < 0 || self.N as isize <= ny || nx < 0 || self.N as isize <= nx {
                                continue;
                            }
                            let ny = ny as usize;
                            let nx = nx as usize;
                            if self.B[ny][nx] != '.' {
                                destruct_cnt += 1;
                            }
                        }
                        candidate.push((destruct_cnt, y, x, i));
                    }
                }
            }
            candidate.sort_by_key(|x| Reverse(x.0));
            let (_, y, x, idx) = candidate[0];
            self.explosion_positions.push((y, x, idx + 1));
            self.destruct_B(y, x, &bombs[idx]);
        }
    }
    fn action(&mut self, bombs: &[Bomb]) {
        let mut py = 0;
        let mut px = 0;
        let mut store_positions = vec![];
        for y in 0..self.N {
            for x in 0..self.N {
                if self.A[y][x] == '@' {
                    store_positions.push((y, x, 0));
                }
            }
        }
        let (idx, a, c) = self.bfs(py, px, &store_positions);
        self.actions.extend(a);
        self.cost += c;
        let (sy, sx, _) = store_positions[idx];
        py = sy;
        px = sx;

        let goal = self.explosion_positions.clone();
        let (idx, _, _) = self.bfs(py, px, &goal);
        let ok = self.check_destruct_A(py, px, &bombs[goal[idx].2 - 1]);
        if ok {
            self.actions.push((2, goal[idx].2));
            self.b += 1;
            self.cost += bombs[goal[idx].2 - 1].C;
        } else {
            for &(_, _, idx) in self.explosion_positions.iter() {
                self.actions.push((2, idx));
                self.b += 1;
                self.cost += bombs[idx - 1].C;
            }
        }

        if ok {
            let (idx, a, c) = self.bfs(py, px, &goal);
            let pos = goal[idx];
            py = pos.0;
            px = pos.1;
            let bomb_idx = pos.2;
            self.explosion_positions.remove(idx);
            self.actions.extend(a);
            self.cost += c;
            // explosion
            self.actions.push((3, bomb_idx));
            self.destruct_A(py, px, &bombs[bomb_idx - 1]);
            self.b -= 1;
        }

        while !self.explosion_positions.is_empty() {
            if self.b == 0 {
                let mut store_positions = vec![];
                for y in 0..self.N {
                    for x in 0..self.N {
                        if self.A[y][x] == '@' {
                            store_positions.push((y, x, 0));
                        }
                    }
                }
                let (idx, a, c) = self.bfs(py, px, &store_positions);
                self.actions.extend(a);
                self.cost += c;
                let (sy, sx, _) = store_positions[idx];
                py = sy;
                px = sx;
            }

            let goal = self.explosion_positions.clone();
            self.b += 1;
            let (idx, a, c) = self.bfs(py, px, &goal);
            let ok = self.check_destruct_A(goal[idx].0, goal[idx].1, &bombs[goal[idx].2 - 1]);
            if ok {
                // buy
                self.actions.push((2, goal[idx].2));
                self.cost += bombs[goal[idx].2 - 1].C;
                let pos = goal[idx];
                py = pos.0;
                px = pos.1;
                let bomb_idx = pos.2;
                self.explosion_positions.remove(idx);
                self.actions.extend(a);
                self.cost += c;
                // explosion
                self.actions.push((3, bomb_idx));
                self.destruct_A(py, px, &bombs[bomb_idx - 1]);
                self.b -= 1;
            } else {
                self.b -= 1;
                for &(_, _, idx) in self.explosion_positions.iter() {
                    self.actions.push((2, idx));
                    self.b += 1;
                    self.cost += bombs[idx - 1].C;
                }
                break;
            }
        }

        while !self.explosion_positions.is_empty() {
            let goal = self.explosion_positions.clone();
            let (idx, a, c) = self.bfs(py, px, &goal);
            // eprintln!("{}", self.cost);
            let pos = goal[idx];
            py = pos.0;
            px = pos.1;
            let bomb_idx = pos.2;
            self.explosion_positions.remove(idx);
            self.actions.extend(a);
            self.cost += c;
            // explosion
            self.actions.push((3, bomb_idx));
            self.destruct_A(py, px, &bombs[bomb_idx - 1]);
            self.b -= 1;
        }
    }
    fn bfs(
        &mut self,
        sy: usize,
        sx: usize,
        goal: &[(usize, usize, usize)],
    ) -> (usize, Vec<(usize, usize)>, usize) {
        let INF = 1_usize << 60;
        let mut dist = vec![vec![INF; self.N]; self.N];
        let mut Q = BinaryHeap::new();
        dist[sy][sx] = 0;
        Q.push((Reverse(dist[sy][sx]), sy, sx));
        while let Some((Reverse(d), py, px)) = Q.pop() {
            if dist[py][px] != d {
                continue;
            }
            for &(dy, dx) in &DIJ4 {
                let ny = py.wrapping_add(dy);
                let nx = px.wrapping_add(dx);
                if ny >= self.N || nx >= self.N {
                    continue;
                }
                let mut c = (self.b + 1) * (self.b + 1);
                if self.A[ny][nx] != '.' {
                    c *= 2;
                }
                if dist[py][px] + c < dist[ny][nx] {
                    dist[ny][nx] = dist[py][px] + c;
                    Q.push((Reverse(dist[ny][nx]), ny, nx))
                }
            }
        }
        let mut candidate = vec![];
        for &(gy, gx, _) in goal.iter() {
            candidate.push((dist[gy][gx], gy, gx));
        }
        candidate.sort();

        let mut mp: HashMap<(usize, usize), char> = HashMap::new();
        mp.insert((1, 0), 'U');
        mp.insert((!0, 0), 'D');
        mp.insert((0, 1), 'L');
        mp.insert((0, !0), 'R');

        let (_, mut gy, mut gx) = candidate[0];
        let mut goal_idx = 0;
        for (i, &(y, x, _)) in goal.iter().enumerate() {
            if gy == y && gx == x {
                goal_idx = i;
                break;
            }
        }
        let mut a = vec![];
        let mut cost = 0;
        while sy != gy || sx != gx {
            for &(dy, dx) in &DIJ4 {
                let ny = gy.wrapping_add(dy);
                let nx = gx.wrapping_add(dx);
                if ny >= self.N || nx >= self.N {
                    continue;
                }
                let mut c = (self.b + 1) * (self.b + 1);
                if self.A[gy][gx] != '.' {
                    c *= 2;
                }
                if dist[ny][nx] + c == dist[gy][gx] {
                    cost += c;
                    a.push((1, mp[&(dy, dx)] as usize));
                    gy = ny;
                    gx = nx;
                    break;
                }
            }
        }
        a.reverse();
        (goal_idx, a, cost)
    }
    fn score(&self) -> usize {
        max!(10, (1e12 as usize) / (1e4 as usize + self.cost))
    }
    fn output(&self) {
        eprintln!("Cost: {}", self.cost);
        eprintln!("Score: {}", self.score());
        println!("{}", self.actions.len());
        for (t, x) in self.actions.iter() {
            if *t == 1 {
                let dir = (*x as u8) as char;
                println!("{} {}", t, dir);
            } else {
                println!("{} {}", t, x);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Bomb {
    C: usize,
    L: usize,
    AB: Vec<(isize, isize)>,
}

impl Bomb {
    fn new(C: usize, L: usize) -> Self {
        input! {
            AB: [(isize, isize); L]
        }
        Bomb { C, L, AB }
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        input! {
            N: usize,
            M: usize,
        }
        let mut state = State::new(N);

        let mut bombs = vec![];
        for _ in 0..M {
            input! {
                C: usize,
                L: usize
            }
            let bomb = Bomb::new(C, L);
            bombs.push(bomb);
        }
        state.get_explosion_positions(&bombs);
        state.action(&bombs);
        state.output();
    }
}

#[macro_export]
macro_rules! max {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::max($x, max!($( $y ),+))
    }
}
#[macro_export]
macro_rules! min {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::min($x, min!($( $y ),+))
    }
}

fn main() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| Solver::default().solve())
        .unwrap()
        .join()
        .unwrap();
}

fn join_to_string<T: std::string::ToString>(v: &[T], sep: &str) -> String {
    v.iter()
        .fold("".to_string(), |s, x| s + x.to_string().as_str() + sep)
}

#[macro_export]
macro_rules! input {
    () => {};
    (mut $var:ident: $t:tt, $($rest:tt)*) => {
        let mut $var = __input_inner!($t);
        input!($($rest)*)
    };
    ($var:ident: $t:tt, $($rest:tt)*) => {
        let $var = __input_inner!($t);
        input!($($rest)*)
    };
    (mut $var:ident: $t:tt) => {
        let mut $var = __input_inner!($t);
    };
    ($var:ident: $t:tt) => {
        let $var = __input_inner!($t);
    };
}

#[macro_export]
macro_rules! __input_inner {
    (($($t:tt),*)) => {
        ($(__input_inner!($t)),*)
    };
    ([$t:tt; $n:expr]) => {
        (0..$n).map(|_| __input_inner!($t)).collect::<Vec<_>>()
    };
    ([$t:tt]) => {{
        let n = __input_inner!(usize);
        (0..n).map(|_| __input_inner!($t)).collect::<Vec<_>>()
    }};
    (chars) => {
        __input_inner!(String).chars().collect::<Vec<_>>()
    };
    (bytes) => {
        __input_inner!(String).into_bytes()
    };
    (usize1) => {
        __input_inner!(usize) - 1
    };
    ($t:ty) => {
        $crate::read::<$t>()
    };
}

#[macro_export]
macro_rules! println {
    () => {
        $crate::write(|w| {
            use std::io::Write;
            std::writeln!(w).unwrap()
        })
    };
    ($($arg:tt)*) => {
        $crate::write(|w| {
            use std::io::Write;
            std::writeln!(w, $($arg)*).unwrap()
        })
    };
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        $crate::write(|w| {
            use std::io::Write;
            std::write!(w, $($arg)*).unwrap()
        })
    };
}

#[macro_export]
macro_rules! flush {
    () => {
        $crate::write(|w| {
            use std::io::Write;
            w.flush().unwrap()
        })
    };
}

pub fn read<T>() -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    use std::cell::RefCell;
    use std::io::*;

    thread_local! {
        pub static STDIN: RefCell<StdinLock<'static>> = RefCell::new(stdin().lock());
    }

    STDIN.with(|r| {
        let mut r = r.borrow_mut();
        let mut s = vec![];
        loop {
            let buf = r.fill_buf().unwrap();
            if buf.is_empty() {
                break;
            }
            if let Some(i) = buf.iter().position(u8::is_ascii_whitespace) {
                s.extend_from_slice(&buf[..i]);
                r.consume(i + 1);
                if !s.is_empty() {
                    break;
                }
            } else {
                s.extend_from_slice(buf);
                let n = buf.len();
                r.consume(n);
            }
        }
        std::str::from_utf8(&s).unwrap().parse().unwrap()
    })
}

pub fn write<F>(f: F)
where
    F: FnOnce(&mut std::io::BufWriter<std::io::StdoutLock>),
{
    use std::cell::RefCell;
    use std::io::*;

    thread_local! {
        pub static STDOUT: RefCell<BufWriter<StdoutLock<'static>>> =
            RefCell::new(BufWriter::new(stdout().lock()));
    }

    STDOUT.with(|w| f(&mut w.borrow_mut()))
}
