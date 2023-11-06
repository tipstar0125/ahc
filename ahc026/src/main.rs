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
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashSet, VecDeque},
    io::{self, BufRead, BufReader},
};

use itertools::Itertools;
use proconio::{
    input,
    marker::{Chars, Usize1},
    source::line::LineSource,
};
use superslice::Ext;

const N: usize = 200;
const M: usize = 10;

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
            elapsed_time * 0.55 >= self.time_threshold
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
            elapsed_time * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    B: Vec<VecDeque<u8>>,
    before_B: HashSet<Vec<VecDeque<u8>>>,
    now: u8,
    cnt: usize,
}

impl State {
    fn new(A: &[Vec<u8>]) -> Self {
        let mut B = vec![];
        for i in 0..M {
            let row: VecDeque<_> = A[i].iter().cloned().collect();
            B.push(row);
        }
        let mut before_B = HashSet::new();
        before_B.insert(B.clone());
        State {
            B,
            before_B,
            now: 1,
            cnt: 0,
        }
    }
    fn move_one(&mut self) {
        let mut tops = vec![0; M];
        for i in 0..M {
            if self.B[i].is_empty() {
                continue;
            }
            let top = *self.B[i].back().unwrap();
            tops[i] = top
        }
        let mut candidate = vec![];
        for i in 0..M {
            if tops[i] == 0 {
                continue;
            }
            for j in i + 1..M {
                if tops[j] == 0 {
                    continue;
                }
                if tops[i] < tops[j] {
                    let diff = tops[j] - tops[i];
                    let small_item = tops[i];
                    let small_box = i;
                    let big_box = j;
                    candidate.push((diff, small_item, small_box, big_box));
                } else if tops[i] > tops[j] {
                    let diff = tops[i] - tops[j];
                    let small_item = tops[j];
                    let small_box = j;
                    let big_box = i;
                    candidate.push((diff, small_item, small_box, big_box));
                }
            }
        }
        candidate.sort();
        for &(_, small_item, small_box, big_box) in &candidate {
            self.B[small_box].pop_back();
            self.B[big_box].push_back(small_item);
            if self.before_B.contains(&self.B) {
                self.B[big_box].pop_back();
                self.B[small_box].push_back(small_item);
                continue;
            }
            self.before_B.insert(self.B.clone());
            println!("{} {}", small_item, big_box + 1);
            break;
        }

        self.cnt += 1;
    }
    fn dig(&mut self) -> bool {
        let mut dig_box = 0;
        for i in 0..M {
            if self.B[i].contains(&self.now) {
                dig_box = i;
            }
        }
        while self.B[dig_box][self.B[dig_box].len() - 1] != self.now {
            let item = *self.B[dig_box].back().unwrap();
            let mut candidate = vec![];
            for i in 0..M {
                if i == dig_box {
                    continue;
                }
                if self.B[i].is_empty() {
                    candidate.push((0, i));
                    continue;
                }
                let mut exists = false;
                for &x in &self.B[i] {
                    if x <= self.now + 1 {
                        exists = true;
                    }
                }
                let pena = if exists { 50 } else { 0 };
                let top = *self.B[i].back().unwrap();
                if item < top {
                    candidate.push(((top - item) as isize + pena, i));
                } else {
                    candidate.push(((item - top) as isize + 15 + pena, i));
                }
            }
            if candidate.is_empty() {
                return false;
            }
            candidate.sort();
            let (_, big_box) = candidate[0];
            self.B[dig_box].pop_back();
            self.B[big_box].push_back(item);
            println!("{} {}", item, big_box + 1);
            self.cnt += 1;
            if self.is_done() {
                break;
            }
        }
        true
    }
    fn separate(&mut self) {
        let mut sep_box = 0;
        for i in 0..M {
            if !self.B[i].is_empty() {
                sep_box = i;
                break;
            }
        }
        let move_box = if (sep_box + 1) == M { 0 } else { sep_box + 1 };
        let idx = self.B[sep_box].len() / 2;
        let item = self.B[sep_box][idx];

        let mut items = vec![];
        while self.B[sep_box][self.B[sep_box].len() - 1] != item {
            items.push(self.B[sep_box].pop_back().unwrap());
        }
        items.push(self.B[sep_box].pop_back().unwrap());

        while let Some(i) = items.pop() {
            self.B[move_box].push_back(i);
        }

        println!("{} {}", item, move_box + 1);
        self.cnt += 1;
    }
    fn remove(&mut self) {
        'outer: loop {
            for i in 0..M {
                if self.B[i].is_empty() {
                    continue;
                }
                let top = *self.B[i].back().unwrap();
                if top == self.now {
                    println!("{} 0", self.now);
                    self.B[i].pop_back();
                    self.now += 1;
                    self.cnt += 1;
                    continue 'outer;
                }
            }
            return;
        }
    }
    fn is_done(&self) -> bool {
        self.now >= N as u8 || self.cnt >= 5000
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        input! {
            _N: usize,
            _M: usize,
            B: [[u8; N / M]; M]
        }

        let mut state = State::new(&B);
        while !state.is_done() {
            let ok = state.dig();
            if !ok {
                state.separate();
            }
            state.remove();
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
