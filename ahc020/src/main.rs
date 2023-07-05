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
use petgraph::visit::Time;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
    source::line,
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
    G: Vec<Vec<(usize, isize, usize)>>,
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
        let mut G = vec![vec![]; *N];
        for (i, &(u, v, w)) in UVW.iter().enumerate() {
            G[u].push((v, w, i));
            G[v].push((u, w, i));
        }

        State {
            P: vec![5000; *N],
            B: vec![1; *M],
            dist_from_station_to_home,
            covered_cnt,
            G,
        }
    }
    fn get_cost_and_score(&self) -> (isize, isize, isize) {
        let power_cost = self.P.iter().map(|x| x * x).sum::<isize>();
        let line_cost = (0..*M)
            .map(|i| self.B[i] as isize * UVW[i].2)
            .sum::<isize>();
        let S = power_cost + line_cost;
        let score = (1e6 * (1.0 + 1e8 / (S as f64 + 1e7))).round() as isize;
        (score, power_cost, line_cost)
    }
    fn greedy_dist_min(&mut self) {
        self.P = vec![0; *N];
        // 各家について、最も近い放送局を探して、電波強度をそれに合わせる貪欲
        for &(a, b) in AB.iter() {
            let mut d_min = INF;
            let mut d_min_station = 0;
            for (i, &(x, y)) in XY.iter().enumerate() {
                let dx = x - a;
                let dy = y - b;
                let d = ((dx * dx) as f64 + (dy * dy) as f64).sqrt().ceil() as isize;
                if d < d_min {
                    d_min = d;
                    d_min_station = i;
                }
            }
            self.P[d_min_station] = max!(self.P[d_min_station], d_min);
        }
        // 各家がカバーされている数を求める
        // 貪欲による初期解生成後、山登りの更新で使用
        self.covered_cnt = vec![0; *K];
        for (i, &(x, y)) in XY.iter().enumerate() {
            for (j, &(a, b)) in AB.iter().enumerate() {
                let dx = x - a;
                let dy = y - b;
                let d = ((dx * dx) as f64 + (dy * dy) as f64).sqrt().ceil() as isize;
                if d <= self.P[i] {
                    self.covered_cnt[j] += 1;
                }
            }
        }
    }
    fn greedy_cost_min(&mut self) {
        self.P = vec![0; *N];
        // 各家について、放送局の電波強度をあげて最小コストになる放送局を探して、電波強度をそれに合わせる貪欲
        for &(a, b) in AB.iter() {
            let mut d_min = INF;
            let mut cost_min_station = 0;
            for (i, &(x, y)) in XY.iter().enumerate() {
                let dx = x - a;
                let dy = y - b;
                let d = ((dx * dx) as f64 + (dy * dy) as f64).sqrt().ceil() as isize;
                let before_d = self.P[i];
                let cost = d * d - before_d * before_d;
                if cost < d_min && d <= P_MAX {
                    d_min = d;
                    cost_min_station = i;
                }
            }
            self.P[cost_min_station] = max!(self.P[cost_min_station], d_min);
        }
        // 各家がカバーされている数を求める
        // 貪欲による初期解生成後、山登りの更新で使用
        self.covered_cnt = vec![0; *K];
        for (i, &(x, y)) in XY.iter().enumerate() {
            for (j, &(a, b)) in AB.iter().enumerate() {
                let dx = x - a;
                let dy = y - b;
                let d = ((dx * dx) as f64 + (dy * dy) as f64).sqrt().ceil() as isize;
                if d <= self.P[i] {
                    self.covered_cnt[j] += 1;
                }
            }
        }
    }
    fn update_covered_cnt(&mut self, station: usize, power: isize) {
        // 放送局の電波強度が更新されたとき、各家がカバーされている数を更新
        // 更新前と比較して、カバーの状態が変化なければ、何もしない
        // 電波強度が放送局と家の距離以上になったら、カバーの数をインクリメント
        // 電波強度が放送局と家の距離より小さくなったら、カバーの数をデクリメント
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
        // 全ての家について、放送局によってカバーされている数が0よりも大きければ、OK
        self.covered_cnt.iter().all(|&x| x > 0)
    }
    fn binary_search_power(&mut self) {
        // 各放送局からの電波の強度を0から順番に二分探索
        // 全ての家がカバーされるギリギリを探す
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
    fn hill_climbing(&mut self, delta: usize, time_keeper: &TimeKeeper) {
        // 少し電波強度を減らして、全ての家をカバーできれば採用
        while !time_keeper.isTimeOver() {
            let station = rnd::gen_range(0, *N);
            let before_power = self.P[station];
            let power = max!(0, before_power - rnd::gen_range(1, delta) as isize);
            self.update_covered_cnt(station, power);
            // カバーできていなければ、不採用として、元に戻す
            if !self.cover_home() {
                self.update_covered_cnt(station, before_power);
            }
        }
    }
    fn annealing(
        &mut self,
        delta: usize,
        time_keeper: &TimeKeeper,
        time_limit: f64,
        start_temp: f64,
        end_temp: f64,
    ) {
        let start_time = time_keeper.get_time();
        let time_limit = time_limit - start_time;
        let mut current_cost = self.P.iter().map(|x| x * x).sum::<isize>();
        let mut cnt = 0;
        eprintln!("Annealing start time: {}", start_time);
        while !time_keeper.isTimeOver() {
            cnt += 1;
            let station = rnd::gen_range(0, *N);
            let before_power = self.P[station];
            if before_power == 0 {
                continue;
            }
            if rnd::gen_bool() {
                // 電波強度を小さくした場合
                let power = max!(0, before_power - rnd::gen_range(1, delta) as isize);
                self.update_covered_cnt(station, power);
                // カバーできていなければ、不採用として、元に戻す
                if !self.cover_home() {
                    self.update_covered_cnt(station, before_power);
                } else {
                    current_cost += power * power - before_power * before_power;
                }
            } else {
                // 電波強度を大きくした場合
                let power = min!(P_MAX, before_power + rnd::gen_range(1, delta) as isize);
                let new_cost = current_cost + power * power - before_power * before_power;
                let T = start_temp
                    + (end_temp - start_temp)
                        * ((time_keeper.get_time() - start_time) / time_limit);
                let prob = ((current_cost as f64 - new_cost as f64) / T).exp();
                if rnd::gen_float() < prob {
                    self.update_covered_cnt(station, power);
                    current_cost = new_cost;
                }
            }
        }
        eprintln!("Annealing count: {}", cnt);
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
    fn partial_kruskal(&mut self) {
        let mut WUV: Vec<_> = UVW
            .iter()
            .zip(0..)
            .map(|((u, v, w), i)| (w, u, v, i))
            .collect();
        WUV.sort();
        let mut stations = vec![];
        for i in 0..*N {
            if self.P[i] > 0 {
                stations.push(i);
            }
        }
        let station_to_node_map: BTreeMap<usize, usize> = stations
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, x)| (x, i))
            .collect();

        let mut uf = UnionFind::new(stations.len());
        for &(_w, u, v, i) in &WUV {
            if stations.contains(u)
                && stations.contains(v)
                && !uf.is_same(station_to_node_map[u], station_to_node_map[v])
            {
                uf.unite(station_to_node_map[u], station_to_node_map[v]);
                self.B[i] = 1;
            } else {
                self.B[i] = 0;
            }
        }
        if uf.get_size() != 1 {
            eprintln!("not partial kruskal");
            self.B = vec![0; *M];
            self.kruskal();
        }
    }
    fn disconnect_no_power_station(&mut self) {
        for _ in 0..100 {
            for i in 1..*N {
                if self.P[i] != 0 {
                    continue;
                }
                for &(_, _, e) in &self.G[i] {
                    if self.B[e] == 0 {
                        continue;
                    }
                    let mut uf = UnionFind::new(*N);
                    self.B[e] = 0;
                    for (j, &b) in self.B.iter().enumerate() {
                        if b == 1 {
                            let (u, v, _) = UVW[j];
                            uf.unite(u, v);
                        }
                    }
                    if uf.get_union_size(i) != 1 {
                        self.B[e] = 1;
                    }
                }
            }
        }
    }
    fn dijkstra(&mut self) {
        self.B = vec![0; *M]; // initialize all off
        let mut d = vec![INF; *N];
        let mut Q = BinaryHeap::new();
        d[0] = 0;
        Q.push(Reverse((0, 0)));
        while !Q.is_empty() {
            let Reverse((_, pos)) = Q.pop().unwrap();
            for &(next, w, _) in &self.G[pos] {
                if d[pos] + w < d[next] {
                    d[next] = d[pos] + w;
                    Q.push(Reverse((d[next], next)));
                }
            }
        }

        for i in 0..*N {
            let mut now = i;
            while now != 0 {
                for &(before, w, e) in &self.G[now] {
                    if d[before] == d[now] - w {
                        now = before;
                        self.B[e] = 1;
                        break;
                    }
                }
            }
        }
        for _ in 0..100 {
            for i in 0..*N {
                if self.P[i] != 0 {
                    continue;
                }
                for &(_, _, e) in &self.G[i] {
                    if self.B[e] == 0 {
                        continue;
                    }
                    let mut uf = UnionFind::new(*N);
                    self.B[e] = 0;
                    for (j, &b) in self.B.iter().enumerate() {
                        if b == 1 {
                            let (u, v, _) = UVW[j];
                            uf.unite(u, v);
                        }
                    }
                    if uf.get_union_size(i) != 1 {
                        self.B[e] = 1;
                    }
                }
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

        let start = std::time::Instant::now();
        let time_limit = 1.8;
        let time_keeper = TimeKeeper::new(time_limit);
        rnd::init(0);

        let mut state = State::new();

        state.greedy_dist_min();
        // state.greedy_cost_min();

        // state.hill_climbing(10, &time_keeper);
        state.annealing(500, &time_keeper, time_limit, 20000.0, 10.0);

        let mut state1 = state.clone();
        let mut state2 = state.clone();

        state1.kruskal();
        state1.disconnect_no_power_station();
        state2.partial_kruskal();
        state2.disconnect_no_power_station();

        let score1 = state1.get_cost_and_score().0;
        let score2 = state2.get_cost_and_score().0;
        if score1 > score2 {
            state = state1;
        } else {
            state = state2;
        }

        let score = state.get_cost_and_score().0;
        eprintln!("Score: {}", score);
        assert!(state.cover_home());

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed time: {}sec", elapsed_time);

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
