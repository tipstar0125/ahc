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
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};
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

const INF: usize = 1_usize << 60;
const TURN: usize = 1e5 as usize;

#[derive(Debug, Clone)]
struct State {
    N: usize,
    G: Vec<Vec<Vec<(usize, usize)>>>,
    d: Vec<Vec<usize>>,
    dirty: Vec<Vec<usize>>,
    last_visited: Vec<Vec<usize>>,
    route: Vec<(usize, usize)>,
    now: (usize, usize),
    actions: Vec<char>,
    visited_cnt: usize,
    turn: usize,
    nears: Vec<Vec<Vec<(usize, usize)>>>,
}

impl State {
    fn read() -> Self {
        input! {
            N: usize,
            h: [Chars; N - 1],
            v: [Chars; N],
            d: [[usize; N]; N]
        }

        let mut G = vec![vec![vec![]; N]; N];

        for i in 0..N {
            for j in 0..N {
                if i + 1 < N && h[i][j] == '0' {
                    G[i][j].push((i + 1, j));
                    G[i + 1][j].push((i, j));
                }
                if j + 1 < N && v[i][j] == '0' {
                    G[i][j].push((i, j + 1));
                    G[i][j + 1].push((i, j));
                }
            }
        }

        let mut last_visited = vec![vec![!0; N]; N];
        let route = vec![(0, 0)];
        last_visited[0][0] = 0;

        let mut nears = vec![vec![vec![]; N]; N];
        for sy in 0..N {
            for sx in 0..N {
                let mut dist = vec![vec![INF; N]; N];
                let mut Q = VecDeque::new();
                Q.push_back((sy, sx));
                dist[sy][sx] = 0;
                while let Some((py, px)) = Q.pop_front() {
                    nears[sy][sx].push((py, px));
                    for &(ny, nx) in &G[py][px] {
                        if dist[py][px] + 1 < dist[ny][nx] {
                            dist[ny][nx] = dist[py][px] + 1;
                            if dist[ny][nx] < 1 {
                                Q.push_back((ny, nx));
                            }
                        }
                    }
                }
            }
        }

        State {
            N,
            G,
            d,
            dirty: vec![vec![0; N]; N],
            last_visited,
            route,
            now: (0, 0),
            actions: vec![],
            visited_cnt: 1,
            turn: 0,
            nears,
        }
    }
    fn get_max_position(&self) -> (usize, usize) {
        let mut my = 0;
        let mut mx = 0;
        let mut max = 0;
        for y in 0..self.N {
            for x in 0..self.N {
                let s = self.dirty[y][x];
                if s >= max {
                    max = self.dirty[y][x];
                    my = y;
                    mx = x;
                }
            }
        }
        (my, mx)
    }
    fn encode_dir(&self, y: usize, x: usize, ny: usize, nx: usize) -> char {
        if y == ny {
            if x > nx {
                'L'
            } else {
                'R'
            }
        } else if y > ny {
            'U'
        } else {
            'D'
        }
    }
    fn decode_dir(&self, y: usize, x: usize, ny: usize, nx: usize) -> char {
        if y == ny {
            if x > nx {
                'R'
            } else {
                'L'
            }
        } else if y > ny {
            'D'
        } else {
            'U'
        }
    }
    fn bfs(&self, goal: Option<(usize, usize)>) -> Vec<char> {
        let (sy, sx) = self.now;
        let (gy, gx) = goal.unwrap_or(self.get_max_position());

        let mut dist = vec![vec![INF; self.N]; self.N];
        dist[sy][sx] = 0;
        let mut Q = VecDeque::new();
        Q.push_back((sy, sx));
        while let Some((py, px)) = Q.pop_front() {
            for &(ny, nx) in &self.G[py][px] {
                if dist[py][px] + 1 < dist[ny][nx] {
                    dist[ny][nx] = dist[py][px] + 1;
                    Q.push_back((ny, nx));
                }
            }
        }

        let mut y = gy;
        let mut x = gx;
        let mut actions = vec![];
        while sy != y || sx != x {
            let mut candidate = vec![];
            for &(ny, nx) in &self.G[y][x] {
                if dist[ny][nx] + 1 != dist[y][x] {
                    continue;
                }
                if self.last_visited[ny][nx] != !0 {
                    candidate.push((0, self.dirty[ny][nx], ny, nx));
                } else {
                    candidate.push((1, self.dirty[ny][nx], ny, nx));
                }
            }
            candidate.sort();
            candidate.reverse();
            let (_, _, ny, nx) = candidate[0];
            let dir = self.decode_dir(y, x, ny, nx);
            actions.push(dir);
            y = ny;
            x = nx;
        }
        actions.reverse();
        actions
    }
    fn near_bfs(&self, destination: &BTreeSet<(usize, usize)>) -> (usize, usize) {
        let (sy, sx) = self.now;

        let mut dist = vec![vec![INF; self.N]; self.N];
        dist[sy][sx] = 0;
        let mut Q = VecDeque::new();
        if destination.contains(&(sy, sx)) {
            return (sy, sx);
        }
        Q.push_back((sy, sx));
        while let Some((py, px)) = Q.pop_front() {
            for &(ny, nx) in &self.G[py][px] {
                if dist[py][px] + 1 < dist[ny][nx] {
                    dist[ny][nx] = dist[py][px] + 1;
                    if destination.contains(&(ny, nx)) {
                        return (ny, nx);
                    }
                    Q.push_back((ny, nx));
                }
            }
        }
        (0, 0)
    }
    fn advance(&mut self, y: &mut usize, x: &mut usize, dir: char) {
        self.last_visited[*y][*x] = self.turn;

        if dir == 'R' {
            *x += 1;
        } else if dir == 'L' {
            *x -= 1;
        } else if dir == 'U' {
            *y -= 1;
        } else {
            *y += 1;
        }

        self.actions.push(dir);
        self.turn += 1;

        if self.last_visited[*y][*x] == !0 {
            self.visited_cnt += 1;
        }

        self.route.push((*y, *x));
    }
    fn update_dirty(&mut self) {
        for y in 0..self.N {
            for x in 0..self.N {
                if self.last_visited[y][x] == !0 {
                    self.dirty[y][x] = self.turn * self.d[y][x];
                } else {
                    self.dirty[y][x] = (self.turn - self.last_visited[y][x]) * self.d[y][x];
                }
            }
        }
        let (y, x) = self.now;
        self.dirty[y][x] = 0;
    }
    fn action(&mut self, goal: Option<(usize, usize)>) {
        let next_actions = self.bfs(goal);
        let (mut y, mut x) = self.now;
        for &dir in &next_actions {
            self.advance(&mut y, &mut x, dir);
        }
        self.now = (y, x);
        self.update_dirty();
    }
    fn action2(&mut self) {
        let (y, x) = self.get_max_position();
        let mut near: BTreeSet<(usize, usize)> = self.nears[y][x].iter().cloned().collect();
        while !near.is_empty() {
            let (ny, nx) = self.near_bfs(&near);
            near.remove(&(ny, nx));
            let next_actions = self.bfs(Some((ny, nx)));
            let (mut y, mut x) = self.now;
            for &dir in &next_actions {
                near.remove(&(y, x));
                self.advance(&mut y, &mut x, dir);
            }
            self.now = (y, x);
        }
        self.update_dirty();
    }
    fn is_visited(&self) -> bool {
        self.visited_cnt >= self.N * self.N
    }
    fn output(&self) {
        println!("{}", self.actions.iter().join(""));
    }
    fn calc_score(&self) -> i64 {
        let L = self.actions.len();
        let mut S = vec![];
        let mut s = 0;
        let mut sum_d = 0;
        for y in 0..self.N {
            for x in 0..self.N {
                s += (L - self.last_visited[y][x]) as i64 * self.d[y][x] as i64;
                sum_d += self.d[y][x] as i64;
            }
        }
        let mut last_visited2 = self.last_visited.clone();
        for t in L..2 * L {
            let (y, x) = self.route[t - L];
            let dt = (t - last_visited2[y][x]) as i64;
            let a = dt * self.d[y][x] as i64;
            s -= a;
            last_visited2[y][x] = t;
            S.push(s);
            s += sum_d;
        }
        let score = (2 * S.iter().sum::<i64>() + L as i64) / (2 * L) as i64;
        score
    }
}

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let start = std::time::Instant::now();

        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            rnd::init(1);
        }

        let state = State::read();
        #[allow(clippy::redundant_clone)]
        let mut best_state = state.clone();
        let mut best_score = 1i64 << 60;
        let mut cnt = 0;

        let time_keeper = TimeKeeper::new(1.8);
        while !time_keeper.isTimeOver() {
            cnt += 1;
            let mut now_state = state.clone();
            let r = rnd::gen_range(50, 90);
            while !now_state.is_visited() {
                if rnd::gen_range(0, 100) <= r {
                    now_state.action2();
                } else {
                    let mut y = 0;
                    let mut x = 0;
                    while now_state.last_visited[y][x] != !0 {
                        y = rnd::gen_range(0, now_state.N);
                        x = rnd::gen_range(0, now_state.N);
                    }
                    now_state.action(Some((y, x)));
                }
            }
            for _ in 0..3 {
                now_state.action2();
            }
            now_state.action(Some((0, 0)));
            let now_score = now_state.calc_score();
            if best_score > now_score && now_state.actions.len() <= TURN {
                best_score = now_score;
                best_state = now_state;
            }
        }

        best_state.output();
        eprintln!("Turn: {} N: {}", best_state.actions.len(), best_state.N);
        eprintln!("cnt: {}", cnt);

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 0.55;
        }
        let score = best_state.calc_score();
        eprintln!("Score: {}", score);
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
