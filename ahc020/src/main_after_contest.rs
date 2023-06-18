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

#[derive(Default)]
struct Solver {}
impl Solver {
    #[fastout]
    fn solve(&mut self) {
        lazy_static::initialize(&_INPUT);

        let start = std::time::Instant::now();
        let time_limit = 1.5;
        let time_keeper = TimeKeeper::new(time_limit);

        let mut P = vec![P_MAX; *N];
        let mut B = vec![0; *M];

        let mut dist = vec![vec![0; *K]; *N];
        let mut to_casting = vec![BTreeSet::new(); *K];
        // 各家について、送られる局を選ぶ
        for (i, &(x, y)) in XY.iter().enumerate() {
            for (j, &(a, b)) in AB.iter().enumerate() {
                let dx = a - x;
                let dy = b - y;
                dist[i][j] = dx * dx + dy * dy;
                if dist[i][j] <= P[i] * P[i] {
                    to_casting[j].insert(i);
                }
            }
        }

        let mut from_casting = vec![BTreeSet::new(); *N];
        // 各家から、最短の局を選び、各局について送る家を格納
        for i in 0..*K {
            let mut min_dist = INF;
            let mut min_station = 0;
            for &v in to_casting[i].iter() {
                if dist[v][i] < min_dist {
                    min_dist = dist[v][i];
                    min_station = v;
                }
            }
            from_casting[min_station].insert(i);
        }

        // 各局の電波を最小限にする
        for i in 0..*N {
            let mut max_dist = 0;
            for &h in from_casting[i].iter() {
                if dist[i][h] > max_dist {
                    max_dist = dist[i][h];
                }
            }
            P[i] = (max_dist as f64).sqrt().ceil() as isize;
        }

        // 電波が出ている局で、電波落としても、届く家が減らなければ、電波を出さないようにする
        for i in 0..*N {
            if P[i] == 0 {
                continue;
            }
            let tmp = P[i];
            P[i] = 0;

            let mut home = BTreeSet::new();
            for j in 0..*N {
                for k in 0..*K {
                    if dist[j][k] <= P[j] * P[j] {
                        home.insert(k);
                    }
                }
            }
            if home.len() < *K {
                P[i] = tmp
            }
        }

        while !time_keeper.isTimeOver() {
            for i in 0..*N {
                let mut from_station_to_home = vec![BTreeSet::new(); *N];
                for i in 0..*N {
                    for j in 0..*K {
                        if dist[i][j] <= P[i] * P[i] {
                            from_station_to_home[i].insert(j);
                        }
                    }
                }
                if P[i] == 0 {
                    continue;
                }
                let mut min_dist_sum = INF;
                let mut min_dist_max = 0;
                let mut min_dist_sum_station = 0;
                for j in 0..*N {
                    if i == j {
                        continue;
                    }
                    let mut from_other_station_to_home = vec![];
                    for &k in from_station_to_home[i].iter() {
                        from_other_station_to_home.push(dist[j][k]);
                    }
                    if from_other_station_to_home.is_empty() {
                        continue;
                    }
                    let s: isize = from_other_station_to_home.iter().sum();
                    if s < min_dist_sum {
                        min_dist_sum = s;
                        min_dist_max = *from_other_station_to_home.iter().max().unwrap();
                        min_dist_sum_station = j;
                    }
                }

                let dist_max_from_other_station_to_home =
                    (min_dist_max as f64).sqrt().ceil() as isize;
                let updated_P = max!(P[min_dist_sum_station], dist_max_from_other_station_to_home);
                if P[i].pow(2) + P[min_dist_sum_station].pow(2) > updated_P.pow(2)
                    && updated_P <= P_MAX
                {
                    P[i] = 0;
                    P[min_dist_sum_station] = updated_P
                }
            }
        }

        let mut G = vec![vec![]; *N];
        for (i, &(u, v, w)) in UVW.iter().enumerate() {
            G[u].push((v, w, i));
            G[v].push((u, w, i));
        }

        dijkstra(&mut B, &G);
        delete_not_used_station(&mut B, &G, &P);

        let mut uf = UnionFind::new(*N);
        for (i, &b) in B.iter().enumerate() {
            if b == 1 {
                let (u, v, _) = UVW[i];
                uf.unite(u, v);
            }
        }
        eprintln!("{:?}", uf.roots);
        eprintln!("{:?}", uf.size);

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed time: {}sec", elapsed_time);

        println!("{}", P.iter().join(" "));
        println!("{}", B.iter().join(" "));
    }
}

fn kruskal(B: &mut [usize]) {
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
            B[i] = 1;
        } else {
            B[i] = 0;
        }
    }
}

fn dijkstra(B: &mut [usize], G: &[Vec<(usize, isize, usize)>]) {
    let mut d = vec![INF; *N];
    let mut Q = BinaryHeap::new();
    d[0] = 0;
    Q.push(Reverse((0, 0)));
    while !Q.is_empty() {
        let Reverse((_, pos)) = Q.pop().unwrap();
        for &(next, w, _) in &G[pos] {
            if d[pos] + w < d[next] {
                d[next] = d[pos] + w;
                Q.push(Reverse((d[next], next)));
            }
        }
    }

    for i in 0..*N {
        let mut now = i;
        while now != 0 {
            for &(before, w, e) in &G[now] {
                if d[before] == d[now] - w {
                    now = before;
                    B[e] = 1;
                    break;
                }
            }
        }
    }
}

fn delete_not_used_station(B: &mut [usize], G: &[Vec<(usize, isize, usize)>], P: &[isize]) {
    for _ in 0..100 {
        for i in 0..*N {
            if P[i] != 0 {
                continue;
            }
            for &(_, _, e) in &G[i] {
                if B[e] == 0 {
                    continue;
                }
                let mut uf = UnionFind::new(*N);
                B[e] = 0;
                for (j, &b) in B.iter().enumerate() {
                    if b == 1 {
                        let (u, v, _) = UVW[j];
                        uf.unite(u, v);
                    }
                }
                if uf.get_union_size(i) != 1 {
                    B[e] = 1;
                }
            }
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
