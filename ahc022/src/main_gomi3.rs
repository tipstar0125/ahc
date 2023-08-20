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
    io::Repeat,
    sync::BarrierWaitResult,
};

use itertools::Itertools;
use num_traits::Pow;
use proconio::{
    fastout, input,
    marker::{Chars, Usize1},
};

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use std::str::Lines;

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
    hot: isize,
    cold: isize,
    dist: Vec<Vec<(isize, isize, isize)>>,
    hot_point: BTreeSet<(usize, usize)>,
}

impl State {
    fn new(
        L: usize,
        N: usize,
        S: usize,
        YX: Vec<(usize, usize)>,
        delta: isize,
        hot_point_num: usize,
    ) -> Self {
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

        let cold = max!(0, MAX / 2 - delta);
        let hot = min!(1000, MAX / 2 + delta);

        let mut P = vec![vec![cold; L]; L];
        let mut hot_point = BTreeSet::new();

        while hot_point.len() < hot_point_num {
            let y = rnd::gen_range(0, L);
            let x = rnd::gen_range(0, L);
            if hot_point.contains(&(y, x)) {
                continue;
            }
            P[y][x] = hot;
            hot_point.insert((y, x));
        }
        assert!(hot_point.len() == hot_point_num);

        let mut dist = vec![];

        for i in 0..N {
            let (y0, x0) = YX[i];
            let mut d = vec![];
            for &(cy, cx) in hot_point.iter() {
                let dy = calc_diff(y0 as isize, cy as isize);
                let dx = calc_diff(x0 as isize, cx as isize);
                d.push((dy.abs() + dx.abs(), dy, dx));
            }
            d.sort();
            dist.push(d);
        }

        State {
            L,
            N,
            S,
            YX,
            P,
            hot,
            cold,
            dist,
            hot_point,
        }
    }
    fn change(&mut self) {
        let calc_diff = |a0: isize, a1: isize| -> isize {
            let diff = a1 - a0;
            if diff.abs() <= self.L as isize / 2 {
                diff
            } else if diff > 0 {
                -(self.L as isize - diff.abs())
            } else {
                self.L as isize - diff.abs()
            }
        };

        let mut P = self.P.clone();
        let mut hot_point = self.hot_point.clone();

        loop {
            let y = rnd::gen_range(0, self.L);
            let x = rnd::gen_range(0, self.L);
            if P[y][x] == self.cold {
                continue;
            }
            P[y][x] = self.cold;
            hot_point.remove(&(y, x));
            loop {
                let ny = rnd::gen_range(0, self.L);
                let nx = rnd::gen_range(0, self.L);
                if P[ny][nx] == self.hot {
                    continue;
                }
                P[ny][nx] = self.hot;
                hot_point.insert((ny, nx));

                let mut dist = vec![];
                for i in 0..self.N {
                    let (y0, x0) = self.YX[i];
                    let mut d = vec![];
                    for &(cy, cx) in hot_point.iter() {
                        let dy = calc_diff(y0 as isize, cy as isize);
                        let dx = calc_diff(x0 as isize, cx as isize);
                        d.push((dy.abs() + dx.abs(), dy, dx));
                    }
                    d.sort();
                    dist.push(d);
                }
                self.dist = dist;
                self.P = P;
                self.hot_point = hot_point;
                break;
            }
            break;
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
        let time_threshold = 3.0;
        let time_keeper = TimeKeeper::new(time_threshold);

        let mut stdin =
            proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
        macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));
        input! {
            L: usize,
            N: usize,
            S: usize,
            YX: [(usize, usize); N]
        }

        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            rnd::init(0);
        }

        input! {
            A: [usize; N],
            F: [isize; 10000]
        }

        let mut rng =
            rand_chacha::ChaCha20Rng::seed_from_u64(rnd::gen_range(0, 1e9 as usize) as u64);
        let mut v_idx = vec![0; N];
        for i in 0..N {
            v_idx[i] = i;
        }
        v_idx.shuffle(&mut rng);
        let mut dummy_A = Vec::new();
        for i in 0..N {
            dummy_A.push(v_idx[i]);
        }

        let mut dummy_F = Vec::new();
        let dist = Normal::<f64>::new(0f64, S as f64).unwrap();
        for _ in 0..10000 {
            let noise = dist.sample(&mut rng);
            let noise = noise.round() as isize;
            dummy_F.push(noise);
        }

        let delta_coefficient = 4;
        let n_coefficient = if S <= 121 { 9 } else { 16 };

        let mut score_max = 0;
        let mut hot_point_num = 1;

        let mut l = 0;
        let mut r = max!(1, min!(1500, L * L));

        let delta = min!(500, delta_coefficient * S as isize);
        let repeat_num = 30;

        'outer: while r - l > 2 {
            let m1 = (2 * l + r) / 3;
            let m2 = (l + 2 * r) / 3;
            let mut score1_sum = 0;
            let mut score2_sum = 0;

            let state1 = State::new(L, N, S, YX.clone(), delta, m1);
            let state2 = State::new(L, N, S, YX.clone(), delta, m2);

            for _ in 0..repeat_num {
                if time_keeper.isTimeOver() {
                    break 'outer;
                }
                let score1 = f(&dummy_A, &dummy_F, m1, n_coefficient, &state1);
                let score2 = f(&dummy_A, &dummy_F, m2, n_coefficient, &state2);
                score1_sum += score1;
                score2_sum += score2;
            }
            if score1_sum > score_max {
                score_max = score1_sum;
                hot_point_num = m1;
            }
            if score2_sum > score_max {
                score_max = score2_sum;
                hot_point_num = m2;
            }

            if score1_sum < score2_sum {
                l = m1;
            } else {
                r = m2;
            }
        }

        for n in l..=r {
            let mut score_sum = 0;
            for _ in 0..repeat_num {
                let state = State::new(L, N, S, YX.clone(), delta, n);
                let score = f(&dummy_A, &dummy_F, n, n_coefficient, &state);
                score_sum += score;
            }
            if score_sum > score_max {
                score_max = score_sum;
                hot_point_num = n;
            }
        }

        eprintln!("Dummy result: {}", score_max / repeat_num);

        let delta = min!(500, delta_coefficient * S as isize);
        eprintln!("delta: {}", delta);
        eprintln!("hot point num: {}", hot_point_num);

        let mut state = State::new(L, N, S, YX, delta, hot_point_num);
        let mut score_max = 0;

        while !time_keeper.isTimeOver() {
            let mut next_state = state.clone();
            next_state.change();
            let score = f(
                &dummy_A,
                &dummy_F,
                hot_point_num,
                n_coefficient,
                &next_state,
            );
            if score > score_max {
                score_max = score;
                state = next_state;
            }
        }

        let place_cost = state.calc_cost();
        eprintln!("Place cost: {}", place_cost);

        for row in &state.P {
            println!("{}", row.iter().join(" "));
        }

        let d = (state.hot - state.cold) as usize;
        let n = n_coefficient * 4 * S * S / (d * d) + 1;
        eprintln!("Measure repeat num: {}", n);

        let mut E = (0..N).collect_vec();
        let mut out_fixed = vec![false; N];
        let mut cnt = 0;
        let mut measure_cost = 0;

        'outer: for i in 0..N {
            let mut candidates = (0..N).collect_vec();
            for p in 0..hot_point_num {
                let mut pattern_out: BTreeMap<(isize, isize, isize), Vec<usize>> = BTreeMap::new();
                for &out in &candidates {
                    if out_fixed[out] {
                        continue;
                    }
                    let (d, dy, dx) = state.dist[out][p];
                    pattern_out.entry((d, dy, dx)).or_default().push(out);
                }

                for &(d, dy, dx) in pattern_out.keys() {
                    let mut measures = vec![];
                    for _ in 0..n {
                        if cnt == 10000 {
                            break 'outer;
                        }

                        println!("{} {} {}", i, dy, dx);

                        let (mut py, mut px) = state.YX[A[i]];
                        let L = state.L as isize;
                        py = ((L + py as isize + dy) % L) as usize;
                        px = ((L + px as isize + dx) % L) as usize;
                        let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                        measure_cost += 100 * (10 + dy.abs() + dx.abs());

                        // input! {
                        //     m: isize
                        // }

                        measures.push(m);
                        cnt += 1;
                    }

                    let sum = measures.iter().sum::<isize>();
                    let ave = sum / n as isize;

                    if (state.hot - ave).abs() < (state.cold - ave).abs() {
                        candidates = pattern_out[&(d, dy, dx)].clone();
                        break;
                    }
                }
                if candidates.len() == 1 {
                    let out = candidates[0];
                    out_fixed[out] = true;
                    E[i] = out;
                    break;
                }
            }
        }

        println!("-1 -1 -1");
        println!("{}", E.iter().join("\n"));

        let mut score = 1e14;
        let mut wrong_answer = 0;
        for i in 0..N {
            if E[i] != A[i] {
                score *= 0.8;
                wrong_answer += 1;
            }
        }

        eprintln!("cnt: {}", cnt);
        eprintln!("Measure cost after: {}", measure_cost);

        score /= place_cost as f64 + measure_cost as f64 + 1e5;
        eprintln!("WA: {}", wrong_answer);
        eprintln!("Score: {}", score.round() as usize);

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

fn f(A: &[usize], F: &[isize], hot_point_num: usize, n_coefficient: usize, state: &State) -> usize {
    let N = state.N;
    let S = state.S;
    let place_cost = state.calc_cost();
    let d = (state.hot - state.cold) as usize;
    let n = n_coefficient * 4 * S * S / (d * d) + 1;

    let mut E = (0..N).collect_vec();
    let mut out_fixed = vec![false; N];
    let mut cnt = 0;
    let mut measure_cost = 0;

    'outer: for i in 0..N {
        let mut candidates = (0..N).collect_vec();
        for p in 0..hot_point_num {
            let mut pattern_out: BTreeMap<(isize, isize, isize), Vec<usize>> = BTreeMap::new();
            for &out in &candidates {
                if out_fixed[out] {
                    continue;
                }
                let (d, dy, dx) = state.dist[out][p];
                pattern_out.entry((d, dy, dx)).or_default().push(out);
            }

            for &(d, dy, dx) in pattern_out.keys() {
                let mut measures = vec![];
                for _ in 0..n {
                    if cnt == 10000 {
                        break 'outer;
                    }

                    let (mut py, mut px) = state.YX[A[i]];
                    let L = state.L as isize;
                    py = ((L + py as isize + dy) % L) as usize;
                    px = ((L + px as isize + dx) % L) as usize;
                    let m = max!(0, min!(1000, state.P[py][px] + F[cnt]));
                    measure_cost += 100 * (10 + dy.abs() + dx.abs());

                    measures.push(m);
                    cnt += 1;
                }

                let sum = measures.iter().sum::<isize>();
                let ave = sum / n as isize;

                if (state.hot - ave).abs() < (state.cold - ave).abs() {
                    candidates = pattern_out[&(d, dy, dx)].clone();
                    break;
                }
            }
            if candidates.len() == 1 {
                let out = candidates[0];
                out_fixed[out] = true;
                E[i] = out;
                break;
            }
        }
    }

    let mut score = 1e14;
    for i in 0..N {
        if E[i] != A[i] {
            score *= 0.8;
        }
    }

    score /= place_cost as f64 + measure_cost as f64 + 1e5;
    score.round() as usize
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
