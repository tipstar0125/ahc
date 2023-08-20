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
    temp_pattern: Vec<usize>,
    hot: isize,
    cold: isize,
}

impl State {
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

        State {
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

        #[cfg(feature = "local")]
        {
            eprintln!("Local Mode");
            rnd::init(1);
        }

        input! {
            A: [usize; N],
            F: [isize; 10000]
        }

        let mut score_max = 0;
        let mut nn = 0;
        let mut dd = 0;

        let mut l = 1;
        let mut r = 20;

        // S=1: score: 90206301, n: 1, d: 10
        // S=4: score: 71838564, n: 1, d: 15
        // S=9: score: 41730584, n: 2, d: 18
        // S=16: score: 26687076, n: 3, d: 24
        // S=25: score: 18042484, n: 3, d: 37
        // S=36: score: 11393926, n: 6, d: 42
        // S=49: score: 7422022, n: 11, d: 46
        // S=64: score: 6378139, n: 13, d: 49
        // S=81: score: 5278563, n: 13, d: 62
        // S=100: score: 4288674, n: 13, d: 76
        // S=121: score: 3401242, n: 13, d: 92
        // S=144: score: 2657447, n: 13, d: 110
        // S=169: score: 2087385, n: 13, d: 129
        // S=196: score: 1653159, n: 13, d: 149
        // S=225: score: 1308516, n: 13, d: 171
        // S=256: score: 992346, n: 13, d: 200
        // S=289: score: 763307, n: 13, d: 231
        // S=324: score: 574686, n: 13, d: 269
        // S=361: score: 473438, n: 13, d: 298
        // S=400: score: 382603, n: 12, d: 297
        // S=441: score: 307194, n: 12, d: 333
        // S=484: score: 249249, n: 12, d: 371
        // S=529: score: 213368, n: 14, d: 401
        // S=576: score: 175761, n: 14, d: 395
        // S=625: score: 144400, n: 14, d: 437
        // S=676: score: 110050, n: 14, d: 448
        // S=729: score: 94609, n: 14, d: 484
        // S=784: score: 46169, n: 14, d: 496
        // S=841: score: 20113, n: 13, d: 481
        // S=900: score: 10050, n: 13, d: 487

        while r - l > 2 {
            eprint!("==================================");
            eprintln!("l: {} r: {}", l, r);
            let n1 = (2 * l + r) / 3;
            let n2 = (l + 2 * r) / 3;
            let (score_n1, d1) = f2(L, N, S, YX.clone(), n1, &A, &F);
            let (score_n2, d2) = f2(L, N, S, YX.clone(), n2, &A, &F);
            if score_n1 > score_max {
                score_max = score_n1;
                nn = n1;
                dd = d1;
            }
            if score_n2 > score_max {
                score_max = score_n2;
                nn = n2;
                dd = d2;
            }

            if score_n1 < score_n2 {
                l = n1;
            } else {
                r = n2;
            }
        }
        for n in l..=r {
            let (score, d) = f2(L, N, S, YX.clone(), n, &A, &F);
            if score > score_max {
                score_max = score;
                nn = n;
                dd = d;
            }
        }
        eprintln!("score: {}, n: {}, d: {}", score_max, nn, dd);
    }
}

fn f(
    L: usize,
    N: usize,
    S: usize,
    YX: Vec<(usize, usize)>,
    n: usize,
    delta: isize,
    A: &Vec<usize>,
    F: &Vec<isize>,
) -> usize {
    let mut pattern_kind = 0;
    let mut nn = N;
    while nn > 0 {
        nn /= 2;
        pattern_kind += 1;
    }
    let mut state = State::new(L, N, S, YX, pattern_kind, delta);
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
    for i in 0..N {
        if E[i] != A[i] {
            score *= 0.8;
            wrong_answer += 1;
        }
    }
    score /= (place_cost as f64 + measure_cost as f64 + 1e5).round();
    eprintln!("WA: {}", wrong_answer);
    eprintln!("Score: {}", score as usize);
    score as usize
}

fn f2(
    L: usize,
    N: usize,
    S: usize,
    YX: Vec<(usize, usize)>,
    n: usize,
    A: &Vec<usize>,
    F: &Vec<isize>,
) -> (usize, isize) {
    let mut l = 10;
    let mut r = 500;
    let mut score_max = 0;
    let mut d = 0;

    while r - l > 2 {
        let d1 = (2 * l + r) / 3;
        let d2 = (l + 2 * r) / 3;
        let score_c1 = f(L, N, S, YX.clone(), n, d1, A, F);
        let score_c2 = f(L, N, S, YX.clone(), n, d2, A, F);
        if score_c1 > score_max {
            score_max = score_c1;
            d = d1;
        }
        if score_c2 > score_max {
            score_max = score_c2;
            d = d2;
        }

        if score_c1 < score_c2 {
            l = d1;
        } else {
            r = d2;
        }
    }

    for c in l..=r {
        let score = f(L, N, S, YX.clone(), n, c, A, F);
        if score > score_max {
            score_max = score;
            d = c;
        }
    }
    (score_max, d)
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
