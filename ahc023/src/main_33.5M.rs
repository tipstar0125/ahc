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

const T: usize = 100;
const H: usize = 20;
const W: usize = 20;
// const T: usize = 10;
// const H: usize = 6;
// const W: usize = 6;
const INF: usize = 1_usize << 60;

#[derive(Debug, Clone)]
struct State {
    entrance: usize,
    G: Vec<Vec<Vec<(usize, usize)>>>,
    SDK: Vec<(usize, usize, usize)>,
    crop: Vec<Vec<(usize, usize)>>,
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
            mut SD: [(usize, usize); K]
        }

        let mut G = vec![vec![vec![]; W]; H];
        let crop = vec![vec![(0, 0); W]; H];

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
        for i in 0..H {
            for j in 0..W {
                if i == entrance && j == 0 {
                    continue;
                }
                let mut visited = vec![vec![false; W]; H];
                let mut Q = VecDeque::new();
                visited[entrance][0] = true;
                Q.push_back((entrance, 0));
                while let Some((now_y, now_x)) = Q.pop_front() {
                    for &(next_y, next_x) in &G[now_y][now_x] {
                        if next_y == i && next_x == j {
                            continue;
                        }
                        if !visited[next_y][next_x] {
                            visited[next_y][next_x] = true;
                            Q.push_back((next_y, next_x));
                        }
                    }
                }
                let mut cnt = 0;
                for ii in 0..H {
                    for jj in 0..W {
                        if visited[ii][jj] {
                            cnt += 1;
                        }
                    }
                }
                if cnt < 395 {
                    ng_sections.insert((i, j));
                    // for &(next_y, next_x) in &G[i][j] {
                    //     ng_sections.insert((next_y, next_x));
                    // }
                }
            }
        }

        let mut SDK = vec![];
        for (i, &(s, d)) in SD.iter().enumerate() {
            SDK.push((s, d, i + 1));
        }
        SDK.sort_by(|(a0, a1, _), (b0, b1, _)| (a0, Reverse(a1)).cmp(&(b0, Reverse(b1))));

        State {
            entrance,
            G,
            SDK,
            crop,
            ng_sections,
        }
    }
    fn plant(&mut self, y: usize, x: usize, s: usize, d: usize) {
        self.crop[y][x] = (s, d);
    }
    fn can_plant_section_list(&self) -> Vec<(usize, usize, usize)> {
        let mut dist = vec![vec![INF; W]; H];
        let mut Q = VecDeque::new();
        Q.push_back((self.entrance, 0));
        dist[self.entrance][0] = 0;
        while let Some((now_y, now_x)) = Q.pop_front() {
            for &(next_y, next_x) in &self.G[now_y][now_x] {
                if dist[now_y][now_x] + 1 < dist[next_y][next_x]
                    && self.crop[next_y][next_x] == (0, 0)
                {
                    dist[next_y][next_x] = dist[now_y][now_x] + 1;
                    Q.push_back((next_y, next_x));
                }
            }
        }
        let mut ret = vec![];
        for i in 0..H {
            for j in 0..W {
                if dist[i][j] < INF {
                    ret.push((self.G[i][j].len(), i, j));
                }
            }
        }
        ret.sort();
        ret
    }
    fn can_plant_section_list2(&self, d: usize) -> Vec<(usize, usize, usize)> {
        let mut dist = vec![vec![INF; W]; H];
        let mut Q = VecDeque::new();
        let mut ret = vec![];
        Q.push_back((self.entrance, 0));
        dist[self.entrance][0] = 0;
        while let Some((now_y, now_x)) = Q.pop_front() {
            for &(next_y, next_x) in &self.G[now_y][now_x] {
                if dist[now_y][now_x] + 1 < dist[next_y][next_x]
                    && self.crop[next_y][next_x] == (0, 0)
                {
                    dist[next_y][next_x] = dist[now_y][now_x] + 1;
                    Q.push_back((next_y, next_x));
                    let mut ok_next_cnt = 4;
                    for &(nn_y, nn_x) in &self.G[next_y][next_x] {
                        if self.crop[nn_y][nn_x].1 < d {
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
                if !visited[next_y][next_x] && self.crop[next_y][next_x].1 <= t {
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
    fn can_harvest_all(&mut self, y: usize, x: usize, s: usize, d: usize) -> bool {
        let mut set_before = BTreeSet::new();
        for &(_, y, x) in &self.can_plant_section_list2(d) {
            set_before.insert((y, x));
        }
        self.plant(y, x, s, d);
        let mut set_after = BTreeSet::new();
        for &(_, y, x) in &self.can_plant_section_list2(d) {
            set_after.insert((y, x));
        }
        if set_before.len() > set_after.len() + 1 {
            return false;
        }

        let mut visited = vec![vec![false; W]; H];
        let mut Q = VecDeque::new();
        visited[self.entrance][0] = true;
        Q.push_back((self.entrance, 0));
        while let Some((now_y, now_x)) = Q.pop_front() {
            for &(next_y, next_x) in &self.G[now_y][now_x] {
                if next_y == y && next_x == x {
                    continue;
                }
                if self.crop[next_y][next_x] != (0, 0) {
                    visited[next_y][next_x] = true;
                    continue;
                }
                if !visited[next_y][next_x] {
                    visited[next_y][next_x] = true;
                    Q.push_back((next_y, next_x));
                }
            }
        }
        visited[y][x] = true;
        Q.push_back((y, x));
        let mut check_list = vec![];
        while let Some((now_y, now_x)) = Q.pop_front() {
            for &(next_y, next_x) in &self.G[now_y][now_x] {
                if !visited[next_y][next_x] {
                    visited[next_y][next_x] = true;
                    if self.crop[next_y][next_x] != (0, 0) {
                        check_list.push((next_y, next_x));
                    }
                    Q.push_back((next_y, next_x));
                }
            }
        }
        let mut ok = true;
        for &(y, x) in &check_list {
            if !self.can_harvest(y, x, self.crop[y][x].1, &set_after) {
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
            rnd::init(1);
        }

        let init_state = State::read();
        let mut best_ans = vec![];
        let mut best_score = 0;
        let time_threshold = 1.8;
        let time_keeper = TimeKeeper::new(time_threshold);

        'outer: loop {
            let mut state = init_state.clone();
            let mut ans = vec![];
            let mut score = 0;
            for &(s, d, k) in &init_state.SDK {
                if time_keeper.isTimeOver() {
                    if score > best_score {
                        best_score = score;
                        best_ans = ans;
                    }
                    break 'outer;
                }
                let mut can_plant_list = state.can_plant_section_list2(d);
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

                let mut cnt = 0;
                let cnt_limit = 5;

                while cnt < cnt_limit {
                    let num = rnd::gen_range(0, S[S.len() - 1]);
                    let idx = S.lower_bound(&num);
                    let (_, y, x) = can_plant_list[idx];
                    if state.ng_sections.contains(&(y, x)) {
                        cnt += 1;
                        continue;
                    }
                    if state.can_harvest_all(y, x, s, d) {
                        ans.push((k, y, x, s));
                        score += d - s + 1;
                        break;
                    } else {
                        state.plant(y, x, 0, 0);
                    }
                    cnt += 1;
                }
                for i in 0..H {
                    for j in 0..W {
                        if state.crop[i][j] == (0, 0) {
                            continue;
                        }
                        if state.crop[i][j].1 < s {
                            state.crop[i][j] = (0, 0);
                        }
                    }
                }
            }
            if score > best_score {
                best_score = score;
                best_ans = ans;
            }
        }

        println!("{}", best_ans.len());
        for row in best_ans {
            println!("{} {} {} {}", row.0, row.1, row.2, row.3);
        }

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        best_score *= 1e6 as usize;
        best_score /= H * W * T;
        eprintln!("Score: {}", best_score);
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
