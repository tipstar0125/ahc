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

const T: usize = 100;
const H: usize = 20;
const W: usize = 20;
const INF: usize = 1_usize << 60;

#[derive(Debug, Clone)]
struct State {
    entrance: usize,
    G: Vec<Vec<Vec<(usize, usize)>>>,
    SDK: Vec<(usize, usize, usize)>,
    crop: Vec<Vec<usize>>,
    crop_plan: Vec<Vec<(usize, Reverse<usize>, usize)>>,
    ng_sections: BTreeSet<(usize, usize)>,
}

impl State {
    fn read() -> Self {
        input! {
            _t: usize,
            _h: usize,
            _w: usize,
            entrance: usize,
            h: [Chars; H - 1],
            v: [Chars; H],
            K: usize,
            SD: [(usize, usize); K]
        }

        let mut G = vec![vec![vec![]; W]; H];
        let crop = vec![vec![0; W]; H];

        for i in 0..H {
            for j in 0..W {
                if i + 1 < H && h[i][j] == '0' {
                    G[i][j].push((i + 1, j));
                    G[i + 1][j].push((i, j));
                }
                if j + 1 < W && v[i][j] == '0' {
                    G[i][j].push((i, j + 1));
                    G[i][j + 1].push((i, j));
                }
            }
        }

        let mut ng_sections = BTreeSet::new();
        ng_sections.insert((entrance, 0));

        let mut SDK = vec![];
        let mut crop_plan = vec![vec![]; T + 1];
        for (i, &(s, d)) in SD.iter().enumerate() {
            SDK.push((s, d, i + 1));
            crop_plan[s].push((d, Reverse(s), i + 1));
        }
        SDK.sort_by(|(a0, a1, _), (b0, b1, _)| (a0, Reverse(a1)).cmp(&(b0, Reverse(b1))));

        State {
            entrance,
            G,
            SDK,
            crop_plan,
            crop,
            ng_sections,
        }
    }
    fn plant(&mut self, y: usize, x: usize, d: usize) {
        self.crop[y][x] = d;
    }
    fn can_plant_section_list(&self, d: usize) -> Vec<(usize, usize, usize)> {
        let mut dist = vec![vec![INF; W]; H];
        let mut Q = VecDeque::new();
        let mut ret = vec![];
        Q.push_back((self.entrance, 0));
        dist[self.entrance][0] = 0;
        while let Some((now_y, now_x)) = Q.pop_front() {
            for &(next_y, next_x) in &self.G[now_y][now_x] {
                if dist[now_y][now_x] + 1 < dist[next_y][next_x] && self.crop[next_y][next_x] == 0 {
                    dist[next_y][next_x] = dist[now_y][now_x] + 1;
                    Q.push_back((next_y, next_x));
                    let mut ok_next_cnt = 4;
                    for &(nn_y, nn_x) in &self.G[next_y][next_x] {
                        if self.crop[nn_y][nn_x] < d {
                            ok_next_cnt -= 1;
                        }
                    }
                    ret.push((ok_next_cnt, next_y, next_x));
                }
            }
        }
        ret.sort();
        ret
    }
    fn can_harvest(&self, sy: usize, sx: usize, t: usize, set: &BTreeSet<(usize, usize)>) -> bool {
        let mut visited = vec![vec![false; W]; H];
        let mut Q = VecDeque::new();
        visited[sy][sx] = true;
        Q.push_back((sy, sx));
        while let Some((now_y, now_x)) = Q.pop_front() {
            for &(next_y, next_x) in &self.G[now_y][now_x] {
                if !visited[next_y][next_x] && self.crop[next_y][next_x] <= t {
                    if set.contains(&(next_y, next_x)) {
                        return true;
                    }
                    visited[next_y][next_x] = true;
                    Q.push_back((next_y, next_x));
                }
            }
        }
        visited[self.entrance][0]
    }
    fn can_harvest_all(&mut self, y: usize, x: usize, d: usize) -> bool {
        let mut set_before = BTreeSet::new();
        for &(_, y, x) in &self.can_plant_section_list(d) {
            set_before.insert((y, x));
        }
        self.plant(y, x, d);
        let mut set_after = BTreeSet::new();
        for &(_, y, x) in &self.can_plant_section_list(d) {
            set_after.insert((y, x));
        }
        if set_before.len() > set_after.len() + 1 {
            return false;
        }

        let mut ok = true;
        for &(ny, nx) in &self.G[y][x] {
            if self.crop[ny][nx] == 0 {
                continue;
            }
            if !self.can_harvest(ny, nx, self.crop[ny][nx], &set_after) {
                ok = false;
            }
        }
        ok
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
            rnd::init(2);
        }

        let mut state = State::read();
        let time_threshold = 2.0 / state.SDK.len() as f64;

        let mut ans = vec![];
        let mut score = 0;
        let mut t = 1;
        let mut not_planted_cnt = 0;
        let window = 2;
        let mut v = vec![];
        for i in 0..window {
            v.extend(state.crop_plan[t + i].clone());
        }
        v.sort();
        while t <= T {
            if v.is_empty() || not_planted_cnt > 20 {
                for i in 0..H {
                    for j in 0..W {
                        if state.crop[i][j] == 0 {
                            continue;
                        }
                        if state.crop[i][j] < t {
                            state.crop[i][j] = 0;
                        }
                    }
                }
                v = v
                    .iter()
                    .filter(|&&(_, Reverse(s), _)| s != t)
                    .cloned()
                    .collect_vec();
                if t + window <= T {
                    v.extend(state.crop_plan[t + window].clone());
                    v.sort();
                }
                t += 1;
                not_planted_cnt = 0;
                continue;
            }
            let (d, Reverse(s), k) = v.pop().unwrap();
            if t == 1 {
                eprintln!("{} {}", s, d);
            }

            let mut can_plant_list = state.can_plant_section_list(d);
            can_plant_list.reverse();
            if can_plant_list.len() <= 1 {
                continue;
            }

            let mut S = vec![];
            let mut sum = 0;
            for i in (0..can_plant_list.len()).rev() {
                sum += i.pow(6);
                S.push(sum);
            }

            let time_keeper = TimeKeeper::new(time_threshold);
            let mut planted = false;

            while !time_keeper.isTimeOver() {
                let num = rnd::gen_range(0, S[S.len() - 1]);
                let idx = S.lower_bound(&num);
                let (_, y, x) = can_plant_list[idx];
                if state.ng_sections.contains(&(y, x)) {
                    continue;
                }
                if state.can_harvest_all(y, x, d) {
                    ans.push((k, y, x, t));
                    score += d - s + 1;
                    planted = true;
                    break;
                } else {
                    state.plant(y, x, 0);
                }
            }
            if !planted {
                not_planted_cnt += 1;
            }
        }

        println!("{}", ans.len());
        for row in ans {
            println!("{} {} {} {}", row.0, row.1, row.2, row.3);
        }

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 0.55;
        }
        score *= 1e6 as usize;
        score /= H * W * T;
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
