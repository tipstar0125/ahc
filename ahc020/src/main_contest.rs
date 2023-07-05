#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
};

use itertools::Itertools;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};

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

        let mut P = vec![5000; *N];
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

        let INF = 1_isize << 60;
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

        let mut G = vec![vec![]; *N];
        for (i, &(u, v, w)) in UVW.iter().enumerate() {
            G[u].push((v, w, i));
            G[v].push((u, w, i));
        }

        // ダイクストラで最短距離を求める
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

        // 経路を復元しながら、使用する辺を選ぶ
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

        // 電波が出ていな局で、無駄な経路を削除（その局だけが削除されるように）
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

        let mut uf = UnionFind::new(*N);
        for (i, &b) in B.iter().enumerate() {
            if b == 1 {
                let (u, v, _) = UVW[i];
                uf.unite(u, v);
            }
        }
        eprintln!("{:?}", uf.roots);
        eprintln!("{:?}", uf.size);

        println!("{}", P.iter().join(" "));
        println!("{}", B.iter().join(" "));
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
