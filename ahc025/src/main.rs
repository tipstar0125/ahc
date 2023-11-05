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
    io::{self, BufRead, BufReader},
};

use itertools::Itertools;
use proconio::{
    input,
    marker::{Chars, Usize1},
    source::line::LineSource,
};
use superslice::Ext;

macro_rules! input(($($tt:tt)*) => (
    let stdin = std::io::stdin();
    let mut stdin = proconio::source::line::LineSource::new(std::io::BufReader::new(stdin));
    proconio::input!(from &mut stdin, $($tt)*);
));

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
    N: usize,
    D: usize,
    Q: usize,
    cnt: usize,
    sorted_index: Vec<usize>,
    sep: Vec<Vec<(usize, usize)>>,
    best_sep: Vec<Vec<(usize, usize)>>,
}

impl State {
    fn new(N: usize, D: usize, Q: usize, cnt: usize, sorted_index: Vec<usize>) -> Self {
        let sep = vec![vec![]; D];
        let best_sep = vec![vec![]; D];
        State {
            N,
            D,
            Q,
            cnt,
            sorted_index,
            sep,
            best_sep,
        }
    }
    fn init(&mut self) {
        for i in 0..self.N {
            let j = self.N - i - 1;
            self.sep[i % self.D].push((j, self.sorted_index[j]));
        }
        self.sep.reverse();
    }
    fn quick_sort(&mut self, A: &[usize]) {
        self.sep = quick_sort2(&self.sep, A, &mut self.cnt, self.Q);
    }
    fn climbing2(&mut self, A: &[usize]) {
        'outer: while self.cnt < self.Q {
            for l in 0..self.D / 2 {
                loop {
                    if self.cnt >= self.Q {
                        break 'outer;
                    }
                    let ans = self.output();
                    println!("#c {}", ans.iter().join(" "));

                    let r = self.D - l - 1;
                    println!(
                        "{} {} {} {}",
                        self.sep[l].len(),
                        self.sep[r].len(),
                        self.sep[l].iter().map(|x| x.1).join(" "),
                        self.sep[r].iter().map(|x| x.1).join(" ")
                    );
                    println!("# {} {}", l, r);
                    if l == self.D / 2 - 1 {
                        println!("# last");
                    }
                    self.cnt += 1;

                    let mut sl = 0;
                    let mut sr = 0;

                    for &(_, idx) in &self.sep[l] {
                        sl += A[idx];
                    }
                    for &(_, idx) in &self.sep[r] {
                        sr += A[idx];
                    }

                    let out = if sl == sr {
                        '='
                    } else if sl < sr {
                        '<'
                    } else {
                        '>'
                    };

                    // input! {
                    //     out: char
                    // }

                    if out == '=' || out == '>' {
                        break;
                    }
                    if self.sep[r].len() <= 1 {
                        break;
                    }
                    self.sep[r].sort();
                    let x = self.sep[r][0];
                    self.sep[r].remove(0);
                    self.sep[l].push(x);
                }
            }
            self.best_sep = self.sep.clone();
            println!("# sort");
            self.quick_sort(A);
        }
    }
    fn climbing(&mut self, A: &[usize]) {
        while self.cnt < self.Q {
            let ans = self.output();
            println!("#c {}", ans.iter().join(" "));

            let mut l = rnd::gen_range(0, self.D);
            let mut r = l;
            while l == r {
                r = rnd::gen_range(0, self.D);
            }

            println!(
                "{} {} {} {}",
                self.sep[l].len(),
                self.sep[r].len(),
                self.sep[l].iter().map(|x| x.1).join(" "),
                self.sep[r].iter().map(|x| x.1).join(" ")
            );
            self.cnt += 1;

            let mut sl = 0;
            let mut sr = 0;

            for &(_, idx) in &self.sep[l] {
                sl += A[idx];
            }
            for &(_, idx) in &self.sep[r] {
                sr += A[idx];
            }

            let out = if sl == sr {
                '='
            } else if sl < sr {
                '<'
            } else {
                '>'
            };

            // input! {
            //     out: char
            // }

            if out == '>' {
                std::mem::swap(&mut l, &mut r);
            }

            if self.sep[r].len() <= 1 {
                continue;
            }

            self.sep[l].sort();
            self.sep[r].sort();

            let x = self.sep[r][0];
            self.sep[r].remove(0);
            self.sep[l].push(x);
        }
    }
    fn is_done(&self) -> bool {
        self.cnt >= self.Q
    }
    fn output(&mut self) -> Vec<usize> {
        // while !self.is_done() {
        //     println!("1 1 0 1");
        //     self.cnt += 1;
        //     // input! {
        //     //     _: char
        //     // }
        // }

        let mut ans = vec![0; self.N];
        for d in 0..self.D {
            for &(_, idx) in &self.best_sep[d] {
                ans[idx] = d;
            }
        }
        ans
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        input! {
            N: usize,
            D: usize,
            Q: usize,
            A: [usize; N]
        }
        // let A = (0..N).collect_vec();

        let mut sorted_index = (0..N).collect_vec();
        let mut cnt = 0;
        sorted_index = quick_sort(&sorted_index, &A, &mut cnt, Q);

        let mut state = State::new(N, D, Q, cnt, sorted_index);
        state.init();
        // state.climbing(&A);
        state.climbing2(&A);
        let ans = state.output();
        println!("{}", ans.iter().join(" "));

        let mut t = vec![0; D];
        for (i, &d) in ans.iter().enumerate() {
            t[d] += A[i] as isize;
        }
        let ave = t.iter().sum::<isize>() / D as isize;
        let mut s = 0;
        for d in 0..D {
            s += (t[d] - ave) * (t[d] - ave);
        }
        let V = (s / D as isize) as f64;
        let score = 1 + (100.0 * V.sqrt()).round() as usize;
        eprintln!("Score: {}", score);
    }
}

fn quick_sort(idx: &Vec<usize>, A: &Vec<usize>, cnt: &mut usize, M: usize) -> Vec<usize> {
    if idx.is_empty() {
        return vec![];
    }
    let x = rnd::gen_range(0, idx.len());
    let mut L = vec![];
    let mut R = vec![];
    for (i, &e) in idx.iter().enumerate() {
        if i == x {
            continue;
        }
        if *cnt + 1 > M {
            if rnd::gen_bool() {
                L.push(e);
            } else {
                R.push(e);
            }
        } else {
            *cnt += 1;
            println!("1 1 {} {}", e, idx[x]);

            let out = if A[e] == A[idx[x]] {
                '='
            } else if A[e] < A[idx[x]] {
                '<'
            } else {
                '>'
            };

            // input! {
            //     out: char
            // }

            if out == '=' {
                if rnd::gen_bool() {
                    L.push(e);
                } else {
                    R.push(e);
                }
            } else if out == '<' {
                L.push(e);
            } else {
                R.push(e);
            }
        }
    }
    L = quick_sort(&L, A, cnt, M);
    R = quick_sort(&R, A, cnt, M);
    L.push(idx[x]);
    L.extend(R);
    L
}

fn quick_sort2(
    idx: &Vec<Vec<(usize, usize)>>,
    A: &[usize],
    cnt: &mut usize,
    M: usize,
) -> Vec<Vec<(usize, usize)>> {
    if idx.is_empty() {
        return vec![];
    }
    let x = rnd::gen_range(0, idx.len());
    let mut L = vec![];
    let mut R = vec![];
    for (i, e) in idx.iter().enumerate() {
        if i == x {
            continue;
        }
        if *cnt + 1 > M {
            if rnd::gen_bool() {
                L.push(e.clone());
            } else {
                R.push(e.clone());
            }
        } else {
            *cnt += 1;
            println!(
                "{} {} {} {}",
                e.len(),
                idx[x].len(),
                e.iter().map(|x| x.1).join(" "),
                idx[x].iter().map(|x| x.1).join(" "),
            );

            let mut sl = 0;
            let mut sr = 0;

            for &(_, idx) in e {
                sl += A[idx];
            }
            for &(_, idx) in &idx[x] {
                sr += A[idx];
            }

            let out = if sl == sr {
                '='
            } else if sl < sr {
                '<'
            } else {
                '>'
            };

            // input! {
            //     out: char
            // }

            if out == '=' {
                if rnd::gen_bool() {
                    L.push(e.clone());
                } else {
                    R.push(e.clone());
                }
            } else if out == '<' {
                L.push(e.clone());
            } else {
                R.push(e.clone());
            }
        }
    }
    L = quick_sort2(&L, A, cnt, M);
    R = quick_sort2(&R, A, cnt, M);
    L.push(idx[x].clone());
    L.extend(R);
    L
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
