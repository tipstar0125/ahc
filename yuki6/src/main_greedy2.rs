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
const NG: isize = -(1 << 60);
const NO_ENEMY: isize = -((1 << 60) + 1);
const CANNOT_BEAT: isize = -((1 << 60) + 2);
const COLLISION: isize = -((1 << 60) + 3);
const THRESHOLD: isize = 300;

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
            field: vec![vec![(0, 0, 0); W]; H + TURN + 10],
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
    fn get_level(&self) -> isize {
        (1 + self.S / 100) as isize
    }
    fn check_front_enemy(&self) -> isize {
        let (now_x, now_y) = self.pos;
        for y in now_y + 1..H + TURN + 10 {
            if self.field[y][now_x] != (0, 0, 0) {
                let (h, p, _) = self.field[y][now_x];
                let dy = y - now_y;
                let level = self.get_level();
                let t = (h + level - 1) / level;
                // t >= 1, dy >= 2
                // dy = 1 => Game over
                if dy <= t as usize {
                    return CANNOT_BEAT;
                } else if level <= THRESHOLD {
                    return p as isize * 1e6 as isize / t;
                } else {
                    return h * 1e6 as isize / t;
                }
            }
        }
        NO_ENEMY
    }
    fn check_left_enemy(&self, d: usize) -> isize {
        let (now_x, now_y) = self.pos;
        let x = (W + now_x - d) % W;
        if self.field[now_y + d][x] != (0, 0, 0) {
            return COLLISION;
        }
        for y in now_y + d + 1..H + TURN + 10 {
            if self.field[y][x] != (0, 0, 0) {
                let (h, p, _) = self.field[y][x];
                let dy = y - now_y;
                let level = self.get_level();
                let t = (h + level - 1) / level;
                if dy <= t as usize {
                    return CANNOT_BEAT;
                } else if level <= THRESHOLD {
                    return p as isize * 1e6 as isize / t;
                } else {
                    return h * 1e6 as isize / t;
                }
            }
        }
        NO_ENEMY
    }
    fn check_right_enemy(&self, d: usize) -> isize {
        let (now_x, now_y) = self.pos;
        let x = (now_x + d) % W;
        if self.field[now_y + d][x] != (0, 0, 0) {
            return COLLISION;
        }
        for y in now_y + d + 1..H + TURN + 10 {
            if self.field[y][x] != (0, 0, 0) {
                let (h, p, _) = self.field[y][x];
                let dy = y - now_y;
                let level = self.get_level();
                let t = (h + level - 1) / level;
                if dy <= t as usize {
                    return CANNOT_BEAT;
                } else if level <= THRESHOLD {
                    return p as isize * 1e6 as isize / t;
                } else {
                    return h * 1e6 as isize / t;
                }
            }
        }
        NO_ENEMY
    }
    fn check_left_num_enemy(&self, d: usize) -> usize {
        let (now_x, now_y) = self.pos;
        let x = (W + now_x - d) % W;
        let mut cnt = 0_usize;
        for y in now_y + d + 1..H + TURN + 10 {
            if self.field[y][x] != (0, 0, 0) {
                cnt += 1;
            }
        }
        cnt
    }
    fn check_right_num_enemy(&self, d: usize) -> usize {
        let (now_x, now_y) = self.pos;
        let x = (now_x + d) % W;
        let mut cnt = 0_usize;
        for y in now_y + d + 1..H + TURN + 10 {
            if self.field[y][x] != (0, 0, 0) {
                cnt += 1;
            }
        }
        cnt
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
            let mut v = vec![];
            v.push((state.check_front_enemy(), 0));
            v.push((state.check_left_enemy(1), -1));
            v.push((state.check_right_enemy(1), 1));
            v.sort();
            v.reverse();
            let (_, action) = v[0];
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
