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

const DIJ4: [(isize, isize); 4] = [(-1, 0), (0, -1), (1, 0), (0, 1)];
const DIJ5: [(isize, isize); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
const DIJ9: [(isize, isize); 9] = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

#[derive(Debug, Clone)]
struct State {
    L: usize,
    N: usize,
    S: usize,
    YX: Vec<(usize, usize)>,
    P: Vec<Vec<isize>>,
    freezed: Vec<Vec<bool>>,
    temp_pattern: Vec<BTreeMap<(isize, isize), isize>>,
}

impl State {
    fn new(L: usize, N: usize, S: usize, YX: Vec<(usize, usize)>, delta: isize) -> Self {
        let cold_limit = max!(0, MAX / 2 - delta) as usize;
        let hot_limit = min!(1000, MAX / 2 + delta) as usize;

        let mut P = vec![vec![0; L]; L];
        for y in 0..L {
            for x in 0..L {
                P[y][x] = rnd::gen_range(cold_limit, hot_limit) as isize;
            }
        }

        let mut temp_pattern = vec![];
        let mut freezed = vec![vec![false; L]; L];
        for i in 0..N {
            let (y, x) = YX[i];
            let mut p = BTreeMap::new();
            for &(dy, dx) in &DIJ5 {
                let ny = ((L as isize + y as isize + dy) % L as isize) as usize;
                let nx = ((L as isize + x as isize + dx) % L as isize) as usize;
                freezed[ny][nx] = true;
                p.insert((dy, dx), P[ny][nx]);
            }
            temp_pattern.push(p);
        }

        State {
            L,
            N,
            S,
            YX,
            P,
            freezed,
            temp_pattern,
        }
    }
    fn climbing(&mut self) {
        for _ in 0..1e7 as usize {
            let y = rnd::gen_range(0, self.L);
            let x = rnd::gen_range(0, self.L);
            if self.freezed[y][x] {
                continue;
            }
            let now_cost = self.calc_single_cost(y, x);
            let now = self.P[y][x];
            let d = rnd::gen_range_isize(3);
            self.P[y][x] = max!(0, min!(MAX, self.P[y][x] + d));
            let next_cost = self.calc_single_cost(y, x);
            if now_cost <= next_cost {
                self.P[y][x] = now;
            }
        }
    }
    fn calc_single_cost(&self, y: usize, x: usize) -> isize {
        let mut cost = 0;
        for (dy, dx) in &DIJ4 {
            let ny = ((self.L as isize + y as isize + dy) % self.L as isize) as usize;
            let nx = ((self.L as isize + x as isize + dx) % self.L as isize) as usize;
            let diff = self.P[y][x] - self.P[ny][nx];
            cost += diff * diff;
        }
        cost
    }
    fn calc_cost(&self) -> isize {
        let mut cost = 0;
        for y in 0..self.L {
            for x in 0..self.L {
                cost += self.calc_single_cost(y, x);
            }
        }
        cost /= 2;
        cost
    }
}

#[derive(Debug, Clone)]
struct State2 {
    L: usize,
    N: usize,
    S: usize,
    YX: Vec<(usize, usize)>,
    P: Vec<Vec<isize>>,
    dist: Vec<(isize, usize, isize, isize)>,
    hot: isize,
    cold: isize,
}

impl State2 {
    fn new(L: usize, N: usize, S: usize, YX: Vec<(usize, usize)>, delta: isize) -> Self {
        let calc_diff = |a0: isize, a1: isize| -> isize {
            let diff = a1 - a0;
            if diff.abs() <= L as isize / 2 {
                diff
            } else if diff > 0 {
                -(L as isize - diff.abs())
            } else {
                L as isize - diff.abs()
            }
        };

        let mut cy = L / 2;
        let mut cx = L / 2;
        let mut d = INF;

        for i in 0..L {
            for j in 0..L {
                let mut s = 0;
                for k in 0..N {
                    let (y, x) = YX[k];
                    let dy = calc_diff(i as isize, y as isize).abs();
                    let dx = calc_diff(j as isize, x as isize).abs();
                    s += dy + dx
                }
                if s < d {
                    d = s;
                    cy = i;
                    cx = j;
                }
            }
        }

        let cold = max!(0, MAX / 2 - delta);
        let hot = min!(1000, MAX / 2 + delta);
        let mut P = vec![vec![cold; L]; L];
        P[cy][cx] = hot;

        let mut dist = vec![];
        for i in 0..N {
            let (y0, x0) = YX[i];
            let dy = calc_diff(y0 as isize, cy as isize);
            let dx = calc_diff(x0 as isize, cx as isize);
            let d = dy.abs() + dx.abs();
            dist.push((d, i, dy, dx));
        }
        dist.sort();

        State2 {
            L,
            N,
            S,
            YX,
            P,
            dist,
            hot,
            cold,
        }
    }
    fn calc_single_cost(&self, y: usize, x: usize) -> isize {
        let mut cost = 0;
        for (dy, dx) in &DIJ4 {
            let ny = ((self.L as isize + y as isize + dy) % self.L as isize) as usize;
            let nx = ((self.L as isize + x as isize + dx) % self.L as isize) as usize;
            let diff = self.P[y][x] - self.P[ny][nx];
            cost += diff * diff;
        }
        cost
    }
    fn calc_cost(&self) -> isize {
        let mut cost = 0;
        for y in 0..self.L {
            for x in 0..self.L {
                cost += self.calc_single_cost(y, x);
            }
        }
        cost /= 2;
        cost
    }
}

#[derive(Debug, Clone)]
struct State3 {
    L: usize,
    N: usize,
    S: usize,
    YX: Vec<(usize, usize)>,
    P: Vec<Vec<isize>>,
    freezed: Vec<Vec<bool>>,
    temp_pattern: Vec<usize>,
    hot: isize,
    cold: isize,
}

impl State3 {
    fn new(
        L: usize,
        N: usize,
        S: usize,
        YX: Vec<(usize, usize)>,
        pattern_kind: usize,
        delta: isize,
    ) -> Self {
        let cold = max!(0, MAX / 2 - delta);
        let hot = min!(1000, MAX / 2 + delta);

        let pattern_num = 1 << pattern_kind;
        let mut used = vec![false; pattern_num];
        let mut P = vec![vec![MAX / 2; L]; L];
        let mut freezed = vec![vec![false; L]; L];

        for i in 0..N {
            let (y, x) = YX[i];
            let mut hot_mask = 0;
            let mut cold_mask = 0;
            for bit in 0..pattern_kind {
                let (dy, dx) = DIJ9[bit];
                let ny = ((L as isize + y as isize + dy) % L as isize) as usize;
                let nx = ((L as isize + x as isize + dx) % L as isize) as usize;
                if P[ny][nx] == hot {
                    hot_mask += 1 << bit;
                }
                if P[ny][nx] == cold {
                    cold_mask += 1 << bit;
                }
            }
            for j in 0..pattern_num {
                if (j & hot_mask == hot_mask) && (!j & cold_mask == cold_mask) && !used[j] {
                    used[j] = true;
                    for bit in 0..pattern_kind {
                        let (dy, dx) = DIJ9[bit];
                        let ny = ((L as isize + y as isize + dy) % L as isize) as usize;
                        let nx = ((L as isize + x as isize + dx) % L as isize) as usize;
                        let temp = if (j >> bit) % 2 == 0 { cold } else { hot };
                        P[ny][nx] = temp;
                        freezed[ny][nx] = true;
                    }
                    break;
                }
            }
        }

        let mut set = BTreeSet::new();
        let mut temp_pattern = vec![];
        for i in 0..N {
            let (y, x) = YX[i];
            let mut pattern = 0_usize;
            for bit in 0..pattern_kind {
                let (dy, dx) = DIJ9[bit];
                let ny = ((L as isize + y as isize + dy) % L as isize) as usize;
                let nx = ((L as isize + x as isize + dx) % L as isize) as usize;
                if P[ny][nx] == hot {
                    pattern += 1 << bit;
                }
            }
            set.insert(pattern);
            temp_pattern.push(pattern);
        }
        eprintln!("{}", set.len());

        State3 {
            L,
            N,
            S,
            YX,
            P,
            freezed,
            temp_pattern,
            hot,
            cold,
        }
    }
    fn climbing(&mut self) {
        for _ in 0..1e7 as usize {
            let y = rnd::gen_range(0, self.L);
            let x = rnd::gen_range(0, self.L);
            if self.freezed[y][x] {
                continue;
            }
            let now_cost = self.calc_single_cost(y, x);
            let now = self.P[y][x];
            let d = rnd::gen_range_isize(5);
            self.P[y][x] = max!(0, min!(MAX, self.P[y][x] + d));
            let next_cost = self.calc_single_cost(y, x);
            if now_cost <= next_cost {
                self.P[y][x] = now;
            }
        }
    }
    fn calc_single_cost(&self, y: usize, x: usize) -> isize {
        let mut cost = 0;
        for (dy, dx) in &DIJ4 {
            let ny = ((self.L as isize + y as isize + dy) % self.L as isize) as usize;
            let nx = ((self.L as isize + x as isize + dx) % self.L as isize) as usize;
            let diff = self.P[y][x] - self.P[ny][nx];
            cost += diff * diff;
        }
        cost
    }
    fn calc_cost(&self) -> isize {
        let mut cost = 0;
        for y in 0..self.L {
            for x in 0..self.L {
                cost += self.calc_single_cost(y, x);
            }
        }
        cost /= 2;
        cost
    }
}

const MAX: isize = 1000;
const INF: isize = 1_isize << 60;

#[derive(Default)]
struct Solver {}
impl Solver {
    fn solve(&mut self) {
        let start = std::time::Instant::now();
        let mut stdin =
            proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
        macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));
        input! {
            L: usize,
            N: usize,
            S: usize,
            YX: [(usize, usize); N]
        }

        // input! {
        //     A: [usize; N],
        //     F: [isize; 10000]
        // }

        let mut delta_mp = BTreeMap::new();
        delta_mp.insert(1, 11);
        delta_mp.insert(4, 37);
        delta_mp.insert(9, 45);
        delta_mp.insert(16, 71);
        delta_mp.insert(25, 77);

        delta_mp.insert(36, 125);
        delta_mp.insert(49, 171);
        delta_mp.insert(64, 223);
        delta_mp.insert(81, 282);
        delta_mp.insert(100, 348);
        delta_mp.insert(121, 421);
        delta_mp.insert(144, 334);
        delta_mp.insert(169, 393);
        delta_mp.insert(196, 455);
        delta_mp.insert(225, 451);
        delta_mp.insert(256, 479);

        delta_mp.insert(400, 297);
        delta_mp.insert(441, 333);
        delta_mp.insert(484, 371);
        delta_mp.insert(529, 401);
        delta_mp.insert(576, 395);
        delta_mp.insert(625, 437);
        delta_mp.insert(676, 448);
        delta_mp.insert(729, 484);
        delta_mp.insert(784, 496);
        delta_mp.insert(841, 481);
        delta_mp.insert(900, 487);
        let delta = if delta_mp.contains_key(&S) {
            delta_mp[&S]
        } else {
            500
        };

        let mut n_mp = BTreeMap::new();
        n_mp.insert(1, 1);
        n_mp.insert(4, 1);
        n_mp.insert(9, 3);
        n_mp.insert(16, 3);
        n_mp.insert(25, 7);

        n_mp.insert(36, 1);
        n_mp.insert(49, 1);
        n_mp.insert(64, 1);
        n_mp.insert(81, 1);
        n_mp.insert(100, 1);
        n_mp.insert(121, 1);
        n_mp.insert(144, 2);
        n_mp.insert(169, 2);
        n_mp.insert(196, 2);
        n_mp.insert(225, 3);
        n_mp.insert(256, 3);

        n_mp.insert(400, 12);
        n_mp.insert(441, 12);
        n_mp.insert(484, 12);
        n_mp.insert(529, 14);
        n_mp.insert(576, 14);
        n_mp.insert(625, 14);
        n_mp.insert(676, 14);
        n_mp.insert(729, 14);
        n_mp.insert(784, 14);
        n_mp.insert(841, 13);
        n_mp.insert(900, 13);

        let n = if n_mp.contains_key(&S) { n_mp[&S] } else { 3 };

        if S <= 25 {
            let mut state = State::new(L, N, S, YX, delta);
            eprintln!("Place cost before: {}", state.calc_cost());
            state.climbing();
            let place_cost = state.calc_cost();
            eprintln!("Place cost after: {}", place_cost);

            for row in &state.P {
                println!("{}", row.iter().join(" "));
            }

            let mut E = (0..N).collect_vec();
            let mut cnt = 0;
            let mut candidates = vec![];
            let mut measure_cost = 0;
            for i in 0..N {
                let mut measure_pattern = BTreeMap::new();
                for &(dy, dx) in &DIJ5 {
                    let mut measures = vec![];
                    for _ in 0..n {
                        println!("{} {} {}", i, dy, dx);

                        // let (mut py, mut px) = state.YX[A[i]];
                        // let L = state.L as isize;
                        // py = ((L + py as isize + dy) % L) as usize;
                        // px = ((L + px as isize + dx) % L) as usize;
                        // let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                        // measure_cost += 100 * (10 + dy.abs() + dx.abs());

                        input! {
                            m: isize
                        }

                        measures.push(m);
                        cnt += 1;
                    }

                    let sum = measures.iter().sum::<isize>();
                    let ave = sum / n as isize;
                    measure_pattern.insert((dy, dx), ave);
                }

                let mut calc_pattern = vec![];
                for (i, p) in state.temp_pattern.iter().enumerate() {
                    let mut diff_pattern = BTreeMap::new();
                    let mut diff_sum = 0;
                    for &(dy, dx) in &DIJ5 {
                        let diff = (p[&(dy, dx)] - measure_pattern[&(dy, dx)]).abs();
                        diff_sum += diff;
                        diff_pattern.insert((dy, dx), diff);
                    }
                    calc_pattern.push((diff_sum, i, diff_pattern));
                }
                calc_pattern.sort();
                candidates.push(calc_pattern);
            }

            let mut score = 1e14;
            for i in 0..N {
                let out = candidates[i][0].1;
                E[i] = out;
                // if out != A[i] {
                //     score *= 0.8;
                // }
            }

            eprintln!("cnt: {}", cnt);
            eprintln!("Measure cost after: {}", measure_cost);

            println!("-1 -1 -1");
            println!("{}", E.iter().join("\n"));

            score /= (place_cost as f64 + measure_cost as f64 + 1e5).round();
            eprintln!("Score: {}", score as usize);
        } else if S <= 361 {
            let state = State2::new(L, N, S, YX, delta);
            let place_cost = state.calc_cost();
            eprintln!("Place cost: {}", place_cost);

            for row in &state.P {
                println!("{}", row.iter().join(" "));
            }

            let mut E = (0..N).collect_vec();

            let mut fixed = vec![false; N];
            let mut cnt = 0;
            let mut measure_cost = 0;

            'outer: for &(_, out, dy, dx) in &state.dist {
                for i in 0..N {
                    if fixed[i] {
                        continue;
                    }

                    let mut measures = vec![];
                    for _ in 0..n {
                        if cnt == 10000 {
                            break 'outer;
                        }
                        println!("{} {} {}", i, dy, dx);

                        // let (mut py, mut px) = state.YX[A[i]];
                        // let L = state.L as isize;
                        // py = ((L + py as isize + dy) % L) as usize;
                        // px = ((L + px as isize + dx) % L) as usize;
                        // let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                        // measure_cost += 100 * (10 + dy.abs() + dx.abs());

                        input! {
                            m: isize
                        }

                        measures.push(m);
                        cnt += 1;
                    }

                    measures.sort();
                    measures.reverse();
                    let sum = measures.iter().sum::<isize>();
                    let ave = sum / n as isize;

                    if (state.hot - ave).abs() < (state.cold - ave).abs() {
                        fixed[i] = true;
                        E[i] = out;
                        break;
                    }
                }
            }

            let mut score = 1e14;
            // for i in 0..N {
            //     if E[i] != A[i] {
            //         score *= 0.8;
            //     }
            // }

            eprintln!("cnt: {}", cnt);
            eprintln!("Measure cost after: {}", measure_cost);

            println!("-1 -1 -1");
            println!("{}", E.iter().join("\n"));

            score /= (place_cost as f64 + measure_cost as f64 + 1e5).round();
            eprintln!("Score: {}", score as usize);
        } else {
            let mut pattern_kind = 0;
            let mut nn = N;
            while nn > 0 {
                nn /= 2;
                pattern_kind += 1;
            }

            let mut state = State3::new(L, N, S, YX, pattern_kind, delta);
            eprintln!("Place cost before: {}", state.calc_cost());
            state.climbing();
            let place_cost = state.calc_cost();
            eprintln!("Place cost after: {}", place_cost);

            for row in &state.P {
                println!("{}", row.iter().join(" "));
            }

            let hot = state.hot;
            let cold = state.cold;

            let mut E = (0..N).collect_vec();
            let mut cnt = 0;
            let mut measure_cost = 0;

            let eval = |measures: &Vec<isize>| -> isize {
                let sum = measures.iter().sum::<isize>();
                let ave = sum / measures.len() as isize;
                if ave < (cold + hot) / 2 {
                    cold
                } else {
                    hot
                }
            };

            'outer: for i in 0..N {
                let mut measured_pattern = 0;
                for bit in 0..pattern_kind {
                    let (dy, dx) = DIJ9[bit];
                    let mut measures = vec![];
                    for _ in 0..n {
                        if cnt == 10000 {
                            break 'outer;
                        }
                        println!("{} {} {}", i, dy, dx);

                        // let (mut py, mut px) = state.YX[A[i]];
                        // let L = state.L as isize;
                        // py = ((L + py as isize + dy) % L) as usize;
                        // px = ((L + px as isize + dx) % L) as usize;
                        // let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                        // measure_cost += 100 * (10 + dy.abs() + dx.abs());

                        input! {
                            m: isize
                        }

                        measures.push(m);
                        cnt += 1;
                    }
                    if eval(&measures) == hot {
                        measured_pattern += 1 << bit;
                    }
                }

                for out in 0..N {
                    if measured_pattern == state.temp_pattern[out] {
                        E[i] = out;
                    }
                }
            }

            eprintln!("Measure count: {}", cnt);
            eprintln!("Measure cost: {}", measure_cost);

            println!("-1 -1 -1");
            println!("{}", E.iter().join("\n"));

            let mut score = 1e14;
            let mut wrong_answer = 0;
            // for i in 0..N {
            //     if E[i] != A[i] {
            //         score *= 0.8;
            //         wrong_answer += 1;
            //     }
            // }

            score /= (place_cost as f64 + measure_cost as f64 + 1e5).round();
            eprintln!("WA: {}", wrong_answer);
            eprintln!("Score: {}", score as usize);
        }

        #[allow(unused_mut, unused_assignments)]
        let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            elapsed_time *= 1.5;
        }
        eprintln!("Elapsed time: {}sec", elapsed_time);
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
