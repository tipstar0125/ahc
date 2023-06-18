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

use itertools::Itertools;
use num_traits::Pow;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};

const INF: isize = 1 << 60;
const P_MAX: isize = 5000;

use lazy_static::lazy_static;

macro_rules! input(($($tt:tt)*) => (
    let stdin = std::io::stdin();
    let mut stdin = proconio::source::line::LineSource::new(stdin.lock());
    proconio::input!(from &mut stdin, $($tt)*);
));

lazy_static! {
    static ref _INPUT: (
        usize,
        usize,
        usize,
        Vec<(isize, isize)>,
        Vec<(usize, usize, isize)>,
        Vec<(isize, isize)>
    ) = {
        input! {
            n: usize,
            m: usize,
            k: usize,
            xy: [(isize, isize); n],
            uvw: [(Usize1, Usize1, isize); m],
            ab: [(isize, isize); k],
        }
        (n, m, k, xy, uvw, ab)
    };
    static ref N: usize = _INPUT.0;
    static ref M: usize = _INPUT.1;
    static ref K: usize = _INPUT.2;
    static ref XY: Vec<(isize, isize)> = _INPUT.3.clone();
    static ref UVW: Vec<(usize, usize, isize)> = _INPUT.4.clone();
    static ref AB: Vec<(isize, isize)> = _INPUT.5.clone();
}

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
    P: Vec<isize>,
    B: Vec<u8>,
    dist_from_station_to_home: Vec<Vec<isize>>,
    covered_cnt: Vec<usize>,
}

impl State {
    fn new() -> Self {
        let mut dist_from_station_to_home = vec![vec![0; *K]; *N];
        let mut covered_cnt = vec![0; *K];

        for (i, &(x, y)) in XY.iter().enumerate() {
            for (j, &(a, b)) in AB.iter().enumerate() {
                let dx = x - a;
                let dy = y - b;
                let d = ((dx * dx) as f64 + (dy * dy) as f64).sqrt().ceil() as isize;
                dist_from_station_to_home[i][j] = d;
                if d <= P_MAX {
                    covered_cnt[j] += 1;
                }
            }
        }

        State {
            P: vec![5000; *N],
            B: vec![1; *M],
            dist_from_station_to_home,
            covered_cnt,
        }
    }
    fn update_covered_cnt(&mut self, station: usize, power: isize) {
        let before_power = self.P[station];
        for (home, d) in self.dist_from_station_to_home[station].iter().enumerate() {
            if (*d <= power) == (*d <= before_power) {
                continue;
            }
            if *d <= power {
                self.covered_cnt[home] += 1;
            } else {
                self.covered_cnt[home] -= 1;
            }
        }
        self.P[station] = power;
    }
    fn cover_home(&self) -> bool {
        self.covered_cnt.iter().all(|&x| x > 0)
    }
    fn binary_search_power(&mut self) {
        for i in 0..*N {
            let mut l = -1;
            let mut r = P_MAX;
            while r - l > 1 {
                let m = (l + r) / 2;
                self.update_covered_cnt(i, m);
                if self.cover_home() {
                    r = m;
                } else {
                    l = m;
                }
            }
            self.update_covered_cnt(i, r);
        }
    }
    fn kruskal(&mut self) {
        let mut WUV: Vec<_> = UVW
            .iter()
            .zip(0..)
            .map(|((u, v, w), i)| (w, u, v, i))
            .collect();
        WUV.sort();
        let mut uf = UnionFind::new(*N);
        for &(_w, u, v, i) in &WUV {
            if !uf.is_same(*u, *v) {
                uf.unite(*u, *v);
                self.B[i] = 1;
            } else {
                self.B[i] = 0;
            }
        }
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    #[fastout]
    fn solve(&mut self) {
        lazy_static::initialize(&_INPUT);

        // let start = std::time::Instant::now();
        // let time_limit = 1.5;
        // let time_keeper = TimeKeeper::new(time_limit);

        let mut state = State::new();
        state.binary_search_power();
        state.kruskal();

        // #[allow(unused_mut, unused_assignments)]
        // let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        // #[cfg(feature = "local")]
        // {
        //     eprintln!("Local Mode");
        //     elapsed_time *= 1.5;
        // }
        // eprintln!("Elapsed time: {}sec", elapsed_time);

        println!("{}", state.P.iter().join(" "));
        println!("{}", state.B.iter().join(" "));
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
