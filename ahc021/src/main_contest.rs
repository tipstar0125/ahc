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
    sync::BarrierWaitResult,
};

use itertools::Itertools;
use num_traits::Pow;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};

const INF: usize = 1 << 60;
const N: usize = 30;
const MAX: usize = 1e4 as usize;

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

#[derive(Debug, Clone)]
struct State {
    B: [[i16; N]; N],
    turn: usize,
}

impl State {
    fn new(B: [[i16; N]; N]) -> Self {
        State { B, turn: 0 }
    }
    fn is_done(&self) -> bool {
        self.turn >= MAX
    }
    fn get_legal_action(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut actions = vec![];
        if x + 1 < N {
            actions.push((x + 1, y));
            actions.push((x + 1, y + 1));
        }
        if x != 0 && self.B[x - 1][y] != -1 {
            actions.push((x - 1, y));
        }
        if x != 0 && y != 0 {
            actions.push((x - 1, y - 1));
        }
        if y != 0 {
            actions.push((x, y - 1));
        }
        if self.B[x][y + 1] != -1 {
            actions.push((x, y + 1));
        }
        actions
    }
    fn swap(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        let tmp = self.B[x0][y0];
        self.B[x0][y0] = self.B[x1][y1];
        self.B[x1][y1] = tmp;
    }
    fn count_error(&self) -> usize {
        let mut cnt = 0;
        for i in 0..N - 1 {
            for j in 0..N {
                if self.B[i][j] == -1 {
                    break;
                }
                let b = self.B[i][j];
                if b > self.B[i + 1][j] || b > self.B[i + 1][j + 1] {
                    cnt += 1;
                }
            }
        }
        cnt
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    #[fastout]
    fn solve(&mut self) {
        let mut B = [[-1_i16; N]; N];
        for i in 0..N {
            input! {
                b: [i16; i+1]
            }
            B[i][..(i + 1)].copy_from_slice(&b[..(i + 1)]);
        }
        let start = std::time::Instant::now();
        let time_limit = 1.9;
        let time_keeper = TimeKeeper::new(time_limit);

        let state = State::new(B);
        let mut ans = vec![];
        let mut cnt = 0;

        let mut best_turn = INF;

        while !time_keeper.isTimeOver() && !state.is_done() {
            cnt += 1;
            let mut actions = vec![];
            let mut num = 0;
            let mut next_state = state.clone();
            while !next_state.is_done() {
                let x = rnd::gen_range(0, N - 1);
                let y = rnd::gen_range(0, x + 1);
                assert!(next_state.B[x][y] != -1);

                let b = next_state.B[x][y];
                if b > next_state.B[x + 1][y] {
                    next_state.B[x][y] = next_state.B[x + 1][y];
                    next_state.B[x + 1][y] = b;
                    next_state.turn += 1;
                    actions.push((x, y, x + 1, y));
                } else if b > next_state.B[x + 1][y + 1] {
                    next_state.B[x][y] = next_state.B[x + 1][y + 1];
                    next_state.B[x + 1][y + 1] = b;
                    next_state.turn += 1;
                    actions.push((x, y, x + 1, y + 1));
                }
                num += 1;
                if num == 2500 {
                    num = 0;
                    if next_state.count_error() == 0 {
                        break;
                    }
                }
            }
            if best_turn > next_state.turn {
                best_turn = next_state.turn;
                ans = actions;
            }
        }
        eprintln!("cnt: {}", cnt);

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed time: {}sec", elapsed_time);

        println!("{}", ans.len());
        for row in ans {
            println!("{} {} {} {}", row.0, row.1, row.2, row.3);
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

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<isize>,
    roots: BTreeSet<usize>,
    size: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let mut roots = BTreeSet::new();
        for i in 0..n {
            roots.insert(i);
        }
        UnionFind {
            parent: vec![-1; n],
            roots,
            size: n,
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] < 0 {
            return x;
        }
        let root = self.find(self.parent[x] as usize);
        self.parent[x] = root as isize;
        root
    }
    fn unite(&mut self, x: usize, y: usize) -> Option<(usize, usize)> {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x == root_y {
            return None;
        }
        let size_x = -self.parent[root_x];
        let size_y = -self.parent[root_y];
        self.size -= 1;
        if size_x >= size_y {
            self.parent[root_x] -= size_y;
            self.parent[root_y] = root_x as isize;
            self.roots.remove(&root_y);
            Some((root_x, root_y))
        } else {
            self.parent[root_y] -= size_x;
            self.parent[root_x] = root_y as isize;
            self.roots.remove(&root_x);
            Some((root_y, root_x))
        }
    }
    fn is_same(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
    fn is_root(&mut self, x: usize) -> bool {
        self.find(x) == x
    }
    fn get_union_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        -self.parent[root] as usize
    }
    fn get_size(&self) -> usize {
        self.size
    }
    fn members(&mut self, x: usize) -> Vec<usize> {
        let root = self.find(x);
        (0..self.parent.len())
            .filter(|i| self.find(*i) == root)
            .collect::<Vec<usize>>()
    }
    fn all_group_members(&mut self) -> BTreeMap<usize, Vec<usize>> {
        let mut groups_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for x in 0..self.parent.len() {
            let r = self.find(x);
            groups_map.entry(r).or_default().push(x);
        }
        groups_map
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
