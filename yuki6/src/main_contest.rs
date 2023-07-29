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

#[derive(Debug, Clone)]
struct State {
    pos: usize,
    S: usize,
    enemy: [[(isize, usize); W]; H],
    turn: usize,
    freeze: bool,
}

impl State {
    fn new() -> Self {
        State {
            pos: 12,
            S: 0,
            enemy: [[(0, 0); W]; H],
            turn: 0,
            freeze: false,
        }
    }
    fn update_enemy(&mut self, HPX: Vec<(usize, usize, usize)>) {
        for i in 1..H {
            self.enemy[i - 1] = self.enemy[i];
        }
        let mut new_enemy = [(0, 0); W];
        for (h, p, x) in HPX {
            new_enemy[x] = (h as isize, p);
        }
        self.enemy[H - 1] = new_enemy;
    }
    fn decide_target(&self) -> Vec<(isize, Reverse<usize>, usize, usize)> {
        let mut targets = vec![];
        for i in 1..H {
            for j in 0..W {
                if self.enemy[i][j] == (0, 0) {
                    continue;
                }
                let (h, p) = self.enemy[i][j];
                if self.pos < j {
                    let right = j - self.pos;
                    let left = self.pos + W - j;
                    if left <= right {
                        let dx = left;
                        let dy = i + 1;
                        if dy as isize - dx as isize - h <= 0 {
                            continue;
                        }
                        targets.push((h, Reverse(p), left, 1));
                    } else {
                        let dx = right;
                        let dy = i + 1;
                        if dy as isize - dx as isize - h <= 0 {
                            continue;
                        }
                        targets.push((h, Reverse(p), right, 2));
                    }
                } else if self.pos > j {
                    let left = self.pos - j;
                    let right = j + W - self.pos;
                    if left <= right {
                        let dx = left;
                        let dy = i + 1;
                        if dy as isize - dx as isize - h <= 0 {
                            continue;
                        }
                        targets.push((h, Reverse(p), left, 1));
                    } else {
                        let dx = right;
                        let dy = i + 1;
                        if dy as isize - dx as isize - h <= 0 {
                            continue;
                        }
                        targets.push((h, Reverse(p), right, 2));
                    }
                } else {
                    let dy = i + 1;
                    if dy as isize - h <= 0 {
                        continue;
                    }
                    targets.push((h, Reverse(p), 0, 0));
                }
            }
        }
        targets.sort();
        targets
    }
    fn update_field(&mut self, dir: usize) {
        if dir == 1 {
            self.pos = (W + self.pos - 1) % W;
        } else if dir == 2 {
            self.pos = (self.pos + 1) % W;
        }
        for i in 1..H {
            if self.enemy[i][self.pos] != (0, 0) {
                let level = (1 + self.S / 100) as isize;
                let (h, p) = self.enemy[i][self.pos];
                if h - level <= 0 {
                    self.S += p;
                    self.enemy[i][self.pos] = (0, 0);
                } else {
                    self.enemy[i][self.pos] = (h - level, p);
                }
                break;
            }
        }
    }
    fn calc_score(&self, dir: usize) -> isize {
        let mut next = self.pos;
        if dir == 1 {
            next = (W + next - 1) % W;
        } else if dir == 2 {
            next = (next + 1) % W;
        }
        if self.enemy[0][next] != (0, 0) {
            return std::isize::MIN;
        } else {
            for i in 1..H {
                if self.enemy[i][next] != (0, 0) {
                    let level = (1 + self.S / 100) as isize;
                    let (h, p) = self.enemy[i][next];
                    if h - level <= 0 {
                        let a = (h - level) / level + 1;
                        return level / a + p as isize * 1000;
                    } else if i == 1 {
                        return std::isize::MIN;
                    } else {
                        let a = (h - level) / level + 1;
                        return level / a;
                    }
                }
            }
        }
        -1
    }
    fn is_done(&self) -> bool {
        self.turn == TURN
    }
    fn output(&self, dir: usize) {
        if dir == 0 {
            println!("S");
        } else if dir == 1 {
            println!("L");
        } else {
            println!("R");
        }
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let mut state = State::new();

        while !state.is_done() {
            let N: isize = read();
            if N == -1 {
                return;
            }

            let mut HPX = vec![];
            for _ in 0..N {
                let v: Vec<usize> = read_vec();
                let h = v[0];
                let p = v[1];
                let x = v[2];
                HPX.push((h, p, x));
            }
            state.update_enemy(HPX);
            let mut scores = vec![];
            for dir in 0..3 {
                scores.push((state.calc_score(dir), Reverse(dir)));
            }
            scores.sort();
            scores.reverse();
            let Reverse(mut dir) = scores[0].1;
            let no_enemy_cnt = scores.iter().filter(|&&x| x.0 == -1).count();
            if no_enemy_cnt == 3 {
                dir = rnd::gen_range(1, 3);
            }
            state.update_field(dir);
            state.turn += 1;
            state.output(dir);
        }
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
