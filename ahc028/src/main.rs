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
};

use itertools::Itertools;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};
use rand_distr::Bernoulli;
use superslice::Ext;

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

const TURN: usize = 5e3 as usize;
const DIJ4: [(usize, usize); 4] = [(!0, 0), (0, !0), (1, 0), (0, 1)];

#[derive(Debug, Clone)]
struct State {
    N: usize,
    M: usize,
    sy: usize,
    sx: usize,
    A: Vec<Vec<u8>>,
    T: Vec<Vec<u8>>,
}

impl State {
    fn new() -> Self {
        input! {
            N: usize,
            M: usize,
            sy: usize,
            sx: usize,
            A: [Chars; N],
            T: [Chars; M]
        }

        let mut AA = vec![vec![0; N]; N];
        for i in 0..N {
            for j in 0..N {
                AA[i][j] = A[i][j] as u8 - b'A';
            }
        }
        let mut TT = vec![vec![0; 5]; M];
        for i in 0..M {
            for j in 0..5 {
                TT[i][j] = T[i][j] as u8 - b'A';
            }
        }

        State {
            N,
            M,
            sy,
            sx,
            A: AA,
            T: TT,
        }
    }
    fn join_string(&mut self) -> Vec<Vec<u8>> {
        let mut TT = self.T.clone();
        loop {
            let mut candidate = vec![];
            for i in 0..TT.len() {
                for j in 0..TT.len() {
                    if i == j {
                        continue;
                    }
                    for k in 0..TT[i].len() {
                        let mut ok = true;
                        for l in 0..TT[i].len() - k {
                            if l >= TT[j].len() {
                                ok = false;
                                break;
                            }
                            if TT[i][k + l] != TT[j][l] {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            candidate.push((k, (i, j)));
                            break;
                        }
                    }
                }
            }
            candidate.sort();
            if !candidate.is_empty() {
                let cnt = candidate[0].0;
                let mut idx1 = candidate[0].1 .0;
                let mut idx2 = candidate[0].1 .1;
                let s = TT[idx1].clone();
                let t = TT[idx2].clone();
                if idx1 < idx2 {
                    std::mem::swap(&mut idx1, &mut idx2);
                }
                TT.remove(idx1);
                TT.remove(idx2);
                let mut st = s[..cnt].to_vec();
                st.extend(t);
                TT.push(st);
            } else {
                break;
            }
        }
        TT
    }
    fn search(&self, y: usize, x: usize, c: u8) -> (usize, usize) {
        let mut Q = VecDeque::new();
        let mut visited = HashSet::new();
        Q.push_back((y, x));
        visited.insert((y, x));
        while let Some((py, px)) = Q.pop_front() {
            if self.A[py][px] == c {
                return (py, px);
            }
            for &(dy, dx) in &DIJ4 {
                let ny = py.wrapping_add(dy);
                let nx = px.wrapping_add(dx);
                if ny >= self.N || nx >= self.N {
                    continue;
                }
                if visited.contains(&(ny, nx)) {
                    continue;
                }
                visited.insert((ny, nx));
                Q.push_back((ny, nx));
            }
        }
        (0, 0)
    }
    fn make_root(&self, t: &[u8]) -> Vec<(usize, usize)> {
        let mut y = rnd::gen_range(0, self.N);
        let mut x = rnd::gen_range(0, self.N);
        while self.A[y][x] != t[0] {
            y = rnd::gen_range(0, self.N);
            x = rnd::gen_range(0, self.N);
        }
        let mut ret = vec![(y, x)];
        for i in 1..t.len() {
            let pos = self.search(y, x, t[i]);
            y = pos.0;
            x = pos.1;
            ret.push((y, x));
        }
        ret
    }
    fn output_string(&self, c: &[u8]) {
        for i in 0..c.len() {
            eprint!("{}", (c[i] + b'A') as char);
        }
        eprintln!();
    }
    fn make_all_roots(&mut self, T: &[Vec<u8>]) -> Vec<(usize, usize)> {
        let mut roots = vec![];
        for i in 0..T.len() {
            let root = self.make_root(&T[i]);
            roots.push(root);
        }

        let mut used = vec![false; roots.len()];
        let mut cnt = 1;
        let idx = rnd::gen_range(0, roots.len());
        used[idx] = true;
        let mut ans = vec![];
        ans.extend(roots[idx].clone());

        while cnt < T.len() {
            let last = ans.last().unwrap();
            let y = last.0;
            let x = last.1;

            let mut candidate = vec![];
            for i in 0..roots.len() {
                if used[i] {
                    continue;
                }
                let (ny, nx) = roots[i][0];
                let d = (ny as isize - y as isize).abs() + (nx as isize - x as isize).abs();
                candidate.push((d, i));
            }
            candidate.sort();
            let idx = candidate[0].1;
            used[idx] = true;
            let root = roots[idx].clone();
            ans.extend(root);
            cnt += 1;
        }
        ans
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let start = std::time::Instant::now();
        let mut state = State::new();
        let time_keeper = TimeKeeper::new(1.9);
        let mut candidate = vec![];
        let T = state.join_string();
        while !time_keeper.isTimeOver() {
            let ans = state.make_all_roots(&T);
            let mut dist = (state.sy as isize - ans[0].0 as isize).abs()
                + (state.sx as isize - ans[0].1 as isize);
            for i in 1..ans.len() {
                let (y0, x0) = ans[i - 1];
                let (y1, x1) = ans[i];
                dist += (y0 as isize - y1 as isize).abs() + (x0 as isize - x1 as isize).abs();
            }
            candidate.push((dist, ans));
        }
        candidate.sort();
        for row in candidate[0].1.iter() {
            println!("{} {}", row.0, row.1);
        }
        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 0.55;
        }
        eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
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
