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
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
};

fn main() {
    let start = std::time::Instant::now();
    #[cfg(feature = "local")]
    {
        let seed = 0;
        eprintln!("Seed: {seed}");
        rnd::init(seed);
    }

    solve();

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

const TARGET: isize = 5e17 as isize;
const TURN_MAX: usize = 50;
const TEMP: usize = 20000;

fn solve() {
    input! {
        N: usize,
        AB: [(isize, isize); N]
    }

    let temp: usize = os_env::get::<usize>("temp").unwrap_or(TEMP);
    let time_keeper = TimeKeeper::new(0.8);

    let mut order = (1..N).collect::<Vec<usize>>();
    let mut best_order = order.clone();
    let mut best_score = play(&AB, &order);
    let mut current_score = best_score;
    let mut iterations = 0;

    while !time_keeper.isTimeOver() {
        iterations += 1;

        let i = rnd::gen_range(0, N - 1);
        let j = rnd::gen_range(0, N - 1);
        if i == j {
            continue;
        }
        order.swap(i, j);

        let score = play(&AB, &order);
        let diff = score as f64 - current_score as f64;
        // diff=score-current_score
        // diff>=0, exp(diff)>=1
        // if rnd::gen_float() < (diff / 1e5 / 0.2).exp() {
        if rnd::gen_float() < (diff / temp as f64).exp() {
            if score > best_score {
                best_score = score;
                best_order = order.clone();
            }
            current_score = score;
        } else {
            order.swap(i, j);
        }
    }

    let mut operations = vec![];
    let score = play(&AB, &best_order);
    for i in 0..N - 1 {
        operations.push((0, best_order[i]));
    }
    output(&operations);
    eprintln!("score: {}", score);
    eprintln!("iterations: {}", iterations);
    eprintln!("turn: {}", operations.len());
}

fn play(AB: &[(isize, isize)], order: &[usize]) -> usize {
    let N = AB.len();
    let mut now = AB[0];
    for i in 0..N - 1 {
        let ret = op(now, AB[order[i]]);
        now = ret;
    }
    calc_score(calc_cost(now))
}

fn op(u: (isize, isize), v: (isize, isize)) -> (isize, isize) {
    ((u.0 + v.0) / 2, (u.1 + v.1) / 2)
}

fn calc_cost(ab: (isize, isize)) -> usize {
    let v1 = (ab.0 - TARGET).abs();
    let v2 = (ab.1 - TARGET).abs();
    max!(v1, v2) as usize
}

fn calc_score(cost: usize) -> usize {
    let score = 2e6 - 1e5 * (cost as f64 + 1.0).log10();
    score.floor() as usize
}

fn output(operations: &[(usize, usize)]) {
    println!("{}", operations.len());
    for &(a, b) in operations.iter() {
        println!("{} {}", a + 1, b + 1);
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

pub mod os_env {
    const PREFIX: &str = "AHC_PARAMS_";

    pub fn get<T: std::str::FromStr>(name: &str) -> Option<T> {
        let name = format!("{}{}", PREFIX, name.to_uppercase());
        std::env::var(name).ok()?.parse().ok()
    }
}

fn join_to_string<T: std::string::ToString>(v: &[T], sep: &str) -> String {
    v.iter()
        .fold("".to_string(), |s, x| s + x.to_string().as_str() + sep)
}

#[macro_export]
macro_rules! input {
    () => {};
    (mut $var:ident: $t:tt, $($rest:tt)*) => {
        let mut $var = __input_inner!($t);
        input!($($rest)*)
    };
    ($var:ident: $t:tt, $($rest:tt)*) => {
        let $var = __input_inner!($t);
        input!($($rest)*)
    };
    (mut $var:ident: $t:tt) => {
        let mut $var = __input_inner!($t);
    };
    ($var:ident: $t:tt) => {
        let $var = __input_inner!($t);
    };
}

#[macro_export]
macro_rules! __input_inner {
    (($($t:tt),*)) => {
        ($(__input_inner!($t)),*)
    };
    ([$t:tt; $n:expr]) => {
        (0..$n).map(|_| __input_inner!($t)).collect::<Vec<_>>()
    };
    ([$t:tt]) => {{
        let n = __input_inner!(usize);
        (0..n).map(|_| __input_inner!($t)).collect::<Vec<_>>()
    }};
    (chars) => {
        __input_inner!(String).chars().collect::<Vec<_>>()
    };
    (bytes) => {
        __input_inner!(String).into_bytes()
    };
    (usize1) => {
        __input_inner!(usize) - 1
    };
    ($t:ty) => {
        $crate::read::<$t>()
    };
}

#[macro_export]
macro_rules! println {
    () => {
        $crate::write(|w| {
            use std::io::Write;
            std::writeln!(w).unwrap()
        })
    };
    ($($arg:tt)*) => {
        $crate::write(|w| {
            use std::io::Write;
            std::writeln!(w, $($arg)*).unwrap()
        })
    };
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        $crate::write(|w| {
            use std::io::Write;
            std::write!(w, $($arg)*).unwrap()
        })
    };
}

#[macro_export]
macro_rules! flush {
    () => {
        $crate::write(|w| {
            use std::io::Write;
            w.flush().unwrap()
        })
    };
}

pub fn read<T>() -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    use std::cell::RefCell;
    use std::io::*;

    thread_local! {
        pub static STDIN: RefCell<StdinLock<'static>> = RefCell::new(stdin().lock());
    }

    STDIN.with(|r| {
        let mut r = r.borrow_mut();
        let mut s = vec![];
        loop {
            let buf = r.fill_buf().unwrap();
            if buf.is_empty() {
                break;
            }
            if let Some(i) = buf.iter().position(u8::is_ascii_whitespace) {
                s.extend_from_slice(&buf[..i]);
                r.consume(i + 1);
                if !s.is_empty() {
                    break;
                }
            } else {
                s.extend_from_slice(buf);
                let n = buf.len();
                r.consume(n);
            }
        }
        std::str::from_utf8(&s).unwrap().parse().unwrap()
    })
}

pub fn write<F>(f: F)
where
    F: FnOnce(&mut std::io::BufWriter<std::io::StdoutLock>),
{
    use std::cell::RefCell;
    use std::io::*;

    thread_local! {
        pub static STDOUT: RefCell<BufWriter<StdoutLock<'static>>> =
            RefCell::new(BufWriter::new(stdout().lock()));
    }

    STDOUT.with(|w| f(&mut w.borrow_mut()))
}
