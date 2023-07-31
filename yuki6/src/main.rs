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
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
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

const H: usize = 60;
const W: usize = 25;
const TURN: usize = 1000;
const INF: usize = 1 << 60;
const NO_ENEMY: isize = -((1 << 60) + 1);
const CANNOT_BEAT: isize = -((1 << 60) + 2);
const COLLISION: isize = -((1 << 60) + 3);

#[derive(Debug, Clone)]
struct State {
    pos: (usize, usize),
    S: usize,
    score: isize,
    field: Vec<Vec<(isize, usize, isize)>>, // hp, power, init hp
    turn: usize,
}

impl State {
    fn new() -> Self {
        State {
            pos: (12, 0),
            S: 0,
            score: 0,
            field: vec![vec![(0, 0, 0); W]; H + TURN + 20],
            turn: 0,
        }
    }
    fn update_field(&mut self, n: usize) {
        let y = H + self.turn;
        for _ in 0..n {
            let v: Vec<usize> = read_vec();
            let h = v[0] as isize;
            let p = v[1];
            let x = v[2];
            self.field[y][x] = (h, p, h);
        }
    }
    fn bfs(&self, depth: usize) -> Vec<(usize, isize)> {
        // (turn, first_action)
        let (now_x, now_y) = self.pos;
        let mut dist = vec![(INF, 0); W];
        let mut Q = VecDeque::new();
        Q.push_back((now_x, 0, 2));
        dist[now_x] = (0, 0);

        while let Some((pos, t, a)) = Q.pop_front() {
            for action in -1..=1 {
                let next_x = ((W as isize + pos as isize + action) as usize) % W;
                let next_y = now_y + t + 1;
                let level = self.get_level();
                let (h, _, _) = self.field[next_y + 1][next_x];
                if t > depth || (self.field[next_y][next_x] != (0, 0, 0)) || h > level {
                    continue;
                }
                if dist[next_x].0 == INF {
                    if a == 2 {
                        dist[next_x] = (dist[pos].0 + 1, action);
                    } else {
                        dist[next_x] = (dist[pos].0 + 1, a);
                    }
                }
                if a == 2 {
                    Q.push_back((next_x, t + 1, action));
                } else {
                    Q.push_back((next_x, t + 1, a));
                }
            }
        }

        dist[now_x].0 += 1;
        dist
    }
    fn bfs2(&self, depth: usize) -> Vec<(usize, isize)> {
        // (turn, first_action)
        let (now_x, now_y) = self.pos;
        let mut dist = vec![vec![(INF, 0); W]; depth + 2];
        let mut Q = VecDeque::new();
        Q.push_back((now_x, 0, 2));
        dist[0][now_x] = (0, 0);

        while let Some((pos, t, a)) = Q.pop_front() {
            for action in -1..=1 {
                let next_x = ((W as isize + pos as isize + action) as usize) % W;
                let next_y = now_y + t + 1;
                let level = self.get_level();
                let (h, _, _) = self.field[next_y + 1][next_x];
                if t > depth || (self.field[next_y][next_x] != (0, 0, 0)) || h > level {
                    continue;
                }
                if dist[t + 1][next_x].0 == INF {
                    if a == 2 {
                        dist[t + 1][next_x] = (dist[t][pos].0 + 1, action);
                    } else {
                        dist[t + 1][next_x] = (dist[t][pos].0 + 1, a);
                    }
                }
                if a == 2 {
                    Q.push_back((next_x, t + 1, action));
                } else {
                    Q.push_back((next_x, t + 1, a));
                }
            }
        }

        dist[0][now_x].0 += 1;
        let mut d = vec![(0, 0); W];
        for j in 0..W {
            let mut turn = INF;
            let mut first_action = 0;
            for i in 0..depth {
                if dist[i][j].0 < turn {
                    turn = dist[i][j].0;
                    first_action = dist[i][j].1;
                }
            }
            d[j] = (turn, first_action);
        }
        d
    }
    fn eval_col(&self) -> isize {
        let col_turn_and_first_action = self.bfs2(8);
        let mut col_eval_result = vec![];
        let (_, now_y) = self.pos;

        'outer: for x in 0..W {
            if col_turn_and_first_action[x].0 == INF {
                continue;
            }
            let (turn, first_action) = col_turn_and_first_action[x];
            let next_y = turn + now_y;
            for y in next_y + 1..H + TURN + 10 {
                if self.field[y][x] != (0, 0, 0) {
                    let (h, p, _) = self.field[y][x];
                    let dy = y - next_y;
                    let level = self.get_level();
                    let t = ((h + level - 1) / level + turn as isize - 1) as usize;
                    if dy > t {
                        let s = p * 1e6 as usize / t;
                        col_eval_result.push((s as isize, first_action));
                    } else {
                        col_eval_result.push((CANNOT_BEAT, first_action));
                        continue 'outer;
                    }
                }
            }
            col_eval_result.push((NO_ENEMY, first_action));
        }
        col_eval_result.sort();
        col_eval_result.reverse();
        col_eval_result[0].1
    }
    fn get_level(&self) -> isize {
        (1 + self.S / 100) as isize
    }
    fn advance(&mut self, action: isize) {
        self.pos.1 += 1;
        self.pos.0 = (W as isize + action + self.pos.0 as isize) as usize % W;
        self.attack();
        self.turn += 1;
        self.output(action);
    }
    fn attack(&mut self) {
        let (now_x, now_y) = self.pos;
        let x = now_x;
        for y in now_y + 1..H + TURN + 10 {
            if self.field[y][x] != (0, 0, 0) {
                let (h, p, init_hp) = self.field[y][x];
                let level = (1 + self.S / 100) as isize;
                if h - level <= 0 {
                    self.S += p;
                    self.score += init_hp;
                    self.field[y][x] = (0, 0, 0);
                } else {
                    self.field[y][x].0 -= level;
                }
                return;
            }
        }
    }
    fn is_done(&self) -> bool {
        self.turn == TURN
    }
    fn output(&self, action: isize) {
        if action == 0 {
            println!("S");
        } else if action == -1 {
            println!("L");
        } else {
            println!("R");
        }
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    #[allow(clippy::vec_init_then_push)]
    fn solve(&mut self) {
        let mut state = State::new();

        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            let _: Vec<usize> = read_vec();
        }

        while !state.is_done() {
            let N: isize = read();
            if N == -1 {
                return;
            }

            state.update_field(N as usize);
            let action = state.eval_col();
            state.advance(action);
        }
        eprintln!("Score: {}", state.score);
        eprintln!("S: {}", state.S);
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

fn read<T: std::str::FromStr>() -> T {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).ok();
    s.trim().parse().ok().unwrap()
}

fn read_vec<T: std::str::FromStr>() -> Vec<T> {
    read::<String>()
        .split_whitespace()
        .map(|e| e.parse().ok().unwrap())
        .collect()
}
