#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(dead_code)]
use std::collections::{vec_deque, BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use itertools::Itertools;
use proconio::{input, marker::*};
use rand::prelude::*;

use svg::node::element::path::Data;
use svg::node::element::{Circle, Group, Line, Path, Rectangle, Style, Text, Title};
use svg::node::Text as TextContent;
use svg::Document;

#[derive(Clone, Debug)]
pub struct Input {
    pub N: usize,
    pub K: usize,
    pub L: usize,
    pub AB: Vec<(usize, usize)>,
    pub C: Vec<Vec<usize>>,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {} {}", self.N, self.K, self.L)?;
        for i in 0..self.K {
            writeln!(f, "{} {}", self.AB[i].0, self.AB[i].1)?;
        }
        for i in 0..self.N {
            writeln!(f, "{}", self.C[i].iter().join(" "))?;
        }
        Ok(())
    }
}

pub fn parse_input(f: &str) -> Input {
    let mut f = proconio::source::once::OnceSource::from(f);
    input! {
        from &mut f,
        N: usize,
        K: usize,
        L: usize,
        AB: [(usize, usize); K],
        C: [[usize; N]; N]
    }
    Input { N, K, L, AB, C }
}

#[derive(Clone, Debug)]
pub struct Output {
    pub out: Vec<usize>,
}

fn read<T: Copy + PartialOrd + std::fmt::Display + std::str::FromStr>(
    token: Option<&str>,
    lb: T,
    ub: T,
) -> Result<T, String> {
    if let Some(v) = token {
        if let Ok(v) = v.parse::<T>() {
            if v < lb || ub < v {
                Err(format!("Out of range: {}", v))
            } else {
                Ok(v)
            }
        } else {
            Err(format!("Parse error: {}", v))
        }
    } else {
        Err("Unexpected EOF".to_owned())
    }
}

pub fn parse_output(input: &Input, f: &str) -> Result<Vec<Output>, String> {
    let mut out = vec![];
    let mut tokens = f.split_whitespace().peekable();
    let L = input.L;
    let K = input.K;
    while tokens.peek().is_some() {
        out.push(read(tokens.next(), 1, L)?);
    }
    if out.len() < K {
        return Err("Too short length".to_owned());
    }
    if out.len() % K != 0 {
        return Err("Lack of length for multi output".to_owned());
    }
    let mut ret = vec![];
    let mut s = vec![];
    for &x in &out {
        s.push(x);
        if s.len() == K {
            ret.push(Output { out: s.clone() });
            s.clear();
        }
    }
    Ok(ret)
}

const MARGIN: f32 = 10.0;

pub fn doc(height: f32, width: f32) -> Document {
    Document::new()
        .set(
            "viewBox",
            (
                -MARGIN,
                -MARGIN,
                width + 2.0 * MARGIN,
                height + 2.0 * MARGIN,
            ),
        )
        .set("width", width + MARGIN)
        .set("height", height + MARGIN)
        .set("style", "background-color:#F2F3F5")
}

pub fn rect(x: f32, y: f32, w: f32, h: f32, fill: &str) -> Rectangle {
    Rectangle::new()
        .set("x", x)
        .set("y", y)
        .set("width", w)
        .set("height", h)
        .set("fill", fill)
}

pub fn cir(x: usize, y: usize, r: usize, fill: &str) -> Circle {
    Circle::new()
        .set("cx", x)
        .set("cy", y)
        .set("r", r)
        .set("fill", fill)
}

pub fn lin(x1: usize, y1: usize, x2: usize, y2: usize, color: &str) -> Line {
    Line::new()
        .set("x1", x1)
        .set("y1", y1)
        .set("x2", x2)
        .set("y2", y2)
        .set("stroke", color)
        .set("stroke-width", 3)
        .set("stroke-linecap", "round")
        .set("stroke-linecap", "round")
        .set("marker-end", "url(#arrowhead)")
    // .set("stroke-dasharray", 5)
}

pub fn arrow(doc: Document) -> Document {
    doc.add(TextContent::new(
        r#"<defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="3" refY="2" orient="auto">
                <polygon points="0 0, 4 2, 0 4" fill="lightgray"/>
            </marker>
        </defs>"#,
    ))
}

pub fn txt(x: usize, y: usize, text: &str) -> Text {
    Text::new()
        .add(TextContent::new(text))
        .set("x", x)
        .set("y", y)
        .set("fill", "black")
}

// 0 <= val <= 1
pub fn color(mut val: f64) -> String {
    val = val.min(1.0);
    val = val.max(0.0);
    let (r, g, b) = if val < 0.5 {
        let x = val * 2.0;
        (
            30. * (1.0 - x) + 144. * x,
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
        )
    } else {
        let x = val * 2.0 - 1.0;
        (
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
            30. * (1.0 - x) + 70. * x,
        )
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        r.round() as i32,
        g.round() as i32,
        b.round() as i32
    )
}

pub fn group(title: String) -> Group {
    Group::new().add(Title::new().add(TextContent::new(title)))
}

pub fn partition(mut doc: Document, h: &[Vec<bool>], v: &[Vec<bool>], size: f32) -> Document {
    let H = v.len();
    let W = h[0].len();
    for i in 0..H + 1 {
        for j in 0..W {
            // Entrance
            // if i == 0 && j == ENTRANCE {
            //     continue;
            // }
            if (i == 0 || i == H) || h[i - 1][j] {
                let data = Data::new()
                    .move_to((size * j as f32, size * i as f32))
                    .line_by((size * 1.0, 0));
                let p = Path::new()
                    .set("d", data)
                    .set("stroke", "black")
                    .set("stroke-width", 3.0)
                    .set("stroke-linecap", "round");
                doc = doc.add(p);
            }
        }
    }
    for j in 0..W + 1 {
        for i in 0..H {
            // Entrance
            // if j == 0 && i == ENTRANCE {
            //     continue;
            // }
            if (j == 0 || j == W) || v[i][j - 1] {
                let data = Data::new()
                    .move_to((size * j as f32, size * i as f32))
                    .line_by((0, size * 1.0));
                let p = Path::new()
                    .set("d", data)
                    .set("stroke", "black")
                    .set("stroke-width", 3.0)
                    .set("stroke-linecap", "round");
                doc = doc.add(p);
            }
        }
    }
    doc
}

pub fn partition_with_outside(
    mut doc: Document,
    h: &[Vec<bool>],
    v: &[Vec<bool>],
    size: f32,
    offset: f32,
) -> Document {
    let H = v.len();
    let W = h[0].len();
    for i in 0..H + 1 {
        for j in 0..W {
            if h[i][j] {
                let data = Data::new()
                    .move_to((size * j as f32 + offset, size * i as f32))
                    .line_by((size * 1.0, 0));
                let p = Path::new()
                    .set("d", data)
                    .set("stroke", "black")
                    .set("stroke-width", 1.0)
                    .set("stroke-dasharray", 2.0)
                    .set("stroke-linecap", "round");
                doc = doc.add(p);
            }
        }
    }
    for j in 0..W + 1 {
        for i in 0..H {
            if v[i][j] {
                let data = Data::new()
                    .move_to((size * j as f32 + offset, size * i as f32))
                    .line_by((0, size * 1.0));
                let p = Path::new()
                    .set("d", data)
                    .set("stroke", "black")
                    .set("stroke-width", 1.0)
                    .set("stroke-dasharray", 2.0)
                    .set("stroke-linecap", "round");
                doc = doc.add(p);
            }
        }
    }
    doc
}

pub fn partition_with_outside_special(
    mut doc: Document,
    h: &[Vec<bool>],
    v: &[Vec<bool>],
    size: f32,
    offset: f32,
) -> Document {
    let H = v.len();
    let W = h[0].len();
    for i in 0..H + 1 {
        for j in 0..W {
            if h[i][j] {
                let data = Data::new()
                    .move_to((size * j as f32 + offset, size * i as f32))
                    .line_by((size * 1.0, 0));
                let p = Path::new()
                    .set("d", data)
                    .set("stroke", "black")
                    .set("stroke-width", 3.0)
                    .set("stroke-linecap", "round");
                doc = doc.add(p);
            }
        }
    }
    for j in 0..W + 1 {
        for i in 0..H {
            if v[i][j] {
                let data = Data::new()
                    .move_to((size * j as f32 + offset, size * i as f32))
                    .line_by((0, size * 1.0));
                let p = Path::new()
                    .set("d", data)
                    .set("stroke", "black")
                    .set("stroke-width", 3.0)
                    .set("stroke-linecap", "round");
                doc = doc.add(p);
            }
        }
    }
    doc
}

pub fn vis(input: &Input, outputs: &[Output], turn: usize) -> String {
    let height = 600.0;
    let width = 1220.0;
    let offset = 620.0;
    let mut doc = doc(height, width);
    let N = input.N;
    let K = input.K;
    let L = input.L;
    let d = height / N as f32;

    // グリッド毎に地区を生成
    for i in 0..N {
        for j in 0..N {
            let k = input.C[i][j];
            let color_area = if k == 0 {
                "white".to_string()
            } else {
                color(k as f64 / K as f64)
            };
            let color_special = if k == 0 {
                "white".to_string()
            } else {
                color(outputs[turn].out[k - 1] as f64 / L as f64)
            };

            let rec_area = rect(j as f32 * d, i as f32 * d, d, d, &color_area)
                .set("stroke", "black")
                .set("stroke-opacity", 0.05)
                .set("fill-opacity", 0.7);
            let rec_special = rect(j as f32 * d + offset, i as f32 * d, d, d, &color_special)
                .set("stroke", "black")
                .set("stroke-opacity", 0.05)
                .set("fill-opacity", 0.7);

            if k == 0 {
                let mut grp_area = group(format!("({},{})", i + 1, j + 1,));
                let mut grp_special = grp_area.clone();
                grp_area = grp_area.add(rec_area);
                grp_special = grp_special.add(rec_special);
                doc = doc.add(grp_area);
                doc = doc.add(grp_special);
            } else {
                let (a, b) = input.AB[k - 1];
                let mut grp_area = group(format!(
                    "({},{})\narea:{}\na:{}\nb:{}\nspecial:{}",
                    i + 1,
                    j + 1,
                    k,
                    a,
                    b,
                    outputs[turn].out[k - 1]
                ));
                let mut grp_special = grp_area.clone();
                grp_area = grp_area.add(rec_area);
                grp_special = grp_special.add(rec_special);
                doc = doc.add(grp_area);
                doc = doc.add(grp_special);
            }
        }
    }

    let DIJ: [(usize, usize); 4] = [(0, 0), (!0, 0), (!0, !0), (0, !0)];

    // 地区毎に多角形（凹凸）のコーナーを検出
    let mut corners = vec![vec![]; K + 1];
    for i in 0..=N {
        for j in 0..=N {
            let mut cnt: HashMap<usize, usize> = HashMap::new();
            for &(di, dj) in &DIJ {
                let ni = i.wrapping_add(di);
                let nj = j.wrapping_add(dj);
                if ni >= N || nj >= N {
                    continue;
                }
                *cnt.entry(input.C[ni][nj]).or_default() += 1;
            }
            for (k, v) in cnt.iter() {
                if *k == 0 {
                    continue;
                }
                if *v == 1 || *v == 3 {
                    corners[*k].push((i, j));
                }
            }
        }
    }

    // コーナー情報から、各地区の外周を破線で囲い、領域範囲を生成
    let mut h = vec![vec![false; N]; N + 1];
    let mut v = vec![vec![false; N + 1]; N];

    for k in 1..=K {
        let mut G: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

        let mut rows: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(i, j) in &corners[k] {
            rows.entry(i).or_default().push(j);
        }
        for (i, js) in rows.iter() {
            assert!(js.len() % 2 == 0);
            let mut c = 0;
            while c < js.len() {
                G.entry((*i, js[c])).or_default().push((*i, js[c + 1]));
                G.entry((*i, js[c + 1])).or_default().push((*i, js[c]));
                c += 2;
            }
        }
        let mut cols: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(i, j) in &corners[k] {
            cols.entry(j).or_default().push(i);
        }
        for (j, is) in cols.iter() {
            assert!(is.len() % 2 == 0);
            let mut c = 0;
            while c < is.len() {
                G.entry((is[c], *j)).or_default().push((is[c + 1], *j));
                G.entry((is[c + 1], *j)).or_default().push((is[c], *j));
                c += 2;
            }
        }

        let mut now = corners[k][0];
        let mut roots = vec![now];
        let mut used = HashSet::new();
        used.insert(now);
        while used.len() < corners[k].len() {
            for next in &G[&now] {
                if !used.contains(next) {
                    used.insert(*next);
                    roots.push(*next);
                    now = *next;
                    break;
                }
            }
        }
        for i in 0..roots.len() {
            let (mut y0, mut x0) = roots[i];
            let (mut y1, mut x1) = roots[(i + 1) % roots.len()];
            if y0 == y1 {
                if x0 > x1 {
                    std::mem::swap(&mut x0, &mut x1);
                }
                for x in x0..x1 {
                    h[y0][x] = true;
                }
            } else {
                if y0 > y1 {
                    std::mem::swap(&mut y0, &mut y1);
                }
                for y in y0..y1 {
                    v[y][x0] = true;
                }
            }
        }
    }
    doc = partition_with_outside(doc, &h, &v, d, 0.0);
    doc = partition_with_outside(doc, &h, &v, d, offset);

    // 特別区の生成
    let mut special = input.C.clone();
    for i in 0..N {
        for j in 0..N {
            if special[i][j] == 0 {
                continue;
            }
            let k = special[i][j];
            special[i][j] = outputs[turn].out[k - 1];
        }
    }

    // 各特別区の外周を実線で囲い、領域範囲を生成
    let mut h = vec![vec![false; N]; N + 1];
    let mut v = vec![vec![false; N + 1]; N];

    for i in 0..=N {
        for j in 0..N {
            if i == 0 {
                if input.C[i][j] != 0 {
                    h[i][j] = true;
                }
            } else if i == N {
                if input.C[i - 1][j] != 0 {
                    h[i][j] = true;
                }
            } else {
                let k = input.C[i][j];
                let bk = input.C[i - 1][j];
                if k == 0 && bk == 0 {
                    continue;
                }
                if k == 0 || bk == 0 || outputs[turn].out[k - 1] != outputs[turn].out[bk - 1] {
                    h[i][j] = true;
                }
            }
        }
    }

    for j in 0..=N {
        for i in 0..N {
            if j == 0 {
                if input.C[i][j] != 0 {
                    v[i][j] = true;
                }
            } else if j == N {
                if input.C[i][j - 1] != 0 {
                    v[i][j] = true;
                }
            } else {
                let k = input.C[i][j];
                let bk = input.C[i][j - 1];
                if k == 0 && bk == 0 {
                    continue;
                }
                if k == 0 || bk == 0 || outputs[turn].out[k - 1] != outputs[turn].out[bk - 1] {
                    v[i][j] = true;
                }
            }
        }
    }

    doc = partition_with_outside_special(doc, &h, &v, d, 0.0);
    doc = partition_with_outside_special(doc, &h, &v, d, offset);
    doc.to_string()
}

const DIJ4: [(usize, usize); 4] = [(!0, 0), (0, !0), (1, 0), (0, 1)];

fn is_connected(input: &Input, output: &Output) -> bool {
    let L = input.L;
    let N = input.N;
    let mut cnt = vec![0; L];
    for i in 0..N {
        for j in 0..N {
            let k = input.C[i][j];
            if k == 0 {
                continue;
            }
            let l = output.out[k - 1];
            cnt[l - 1] += 1;
        }
    }
    for l in 0..L {
        let mut si = 0;
        let mut sj = 0;
        for i in 0..N {
            for j in 0..N {
                let k = input.C[i][j];
                if k == 0 {
                    continue;
                }
                if l == output.out[k - 1] - 1 {
                    si = i;
                    sj = j;
                }
            }
        }
        let mut c = 1;
        let mut visited = vec![vec![false; N]; N];
        let mut Q = VecDeque::new();
        visited[si][sj] = true;
        Q.push_back((si, sj));
        while let Some((pi, pj)) = Q.pop_front() {
            for &(di, dj) in &DIJ4 {
                let ni = pi.wrapping_add(di);
                let nj = pj.wrapping_add(dj);
                if ni >= N || nj >= N {
                    continue;
                }
                if visited[ni][nj] {
                    continue;
                }
                let nk = input.C[ni][nj];
                if nk == 0 {
                    continue;
                }
                if l != output.out[nk - 1] - 1 {
                    continue;
                }
                visited[ni][nj] = true;
                c += 1;
                Q.push_back((ni, nj));
            }
        }
        if cnt[l] != c {
            return false;
        }
    }
    true
}

pub fn compute_score(input: &Input, outputs: &[Output], turn: usize) -> (i64, String) {
    let st: HashSet<usize> = outputs[turn].out.clone().into_iter().collect();
    if st.len() != input.L {
        return (
            0,
            "There are special wards that are not assigned to any district".to_string(),
        );
    }
    let K = input.K;
    let L = input.L;
    let mut specials = vec![vec![]; L];
    for k in 0..K {
        let l = outputs[turn].out[k] - 1;
        specials[l].push(k);
    }
    let mut pp = vec![];
    let mut qq = vec![];
    for l in 0..L {
        let mut p = 0;
        let mut q = 0;
        for k in specials[l].iter() {
            p += input.AB[*k].0;
            q += input.AB[*k].1;
        }
        pp.push(p);
        qq.push(q);
    }
    let pmax = *pp.iter().max().unwrap() as f64;
    let pmin = *pp.iter().min().unwrap() as f64;
    let qmax = *qq.iter().max().unwrap() as f64;
    let qmin = *qq.iter().min().unwrap() as f64;
    let mut score = (pmin / pmax).min(qmin / qmax);
    if is_connected(input, &outputs[turn]) {
        score *= 1e6;
    } else {
        score *= 1e3;
    }
    let score = score.round() as i64;
    (score, String::new())
}

pub fn gen(seed: u64) -> Input {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    let N = 50;
    let K = 400;
    let L = 20;
    let mut AB = vec![];
    for _ in 0..K {
        let a = rng.gen_range(50000, 100001);
        let b = rng.gen_range(1000, 2001);
        AB.push((a, b));
    }

    let mut C = vec![vec![0; N]; N];
    let J = (1.2 * K as f64).round() as usize;
    let mut v = (0..N * N).collect_vec();
    v.shuffle(&mut rng);
    for i in 0..N {
        for j in 0..N {
            C[i][j] = v[i * N + j];
        }
    }

    let mut used = vec![vec![false; N]; N];
    let mut c = 0;
    let mut cnt = 0;
    while c < N * N {
        let i = rng.gen_range(0, N);
        let j = rng.gen_range(0, N);
        if used[i][j] {
            continue;
        }
        used[i][j] = true;
        c += 1;
        let mut cand = vec![];
        for &(di, dj) in &DIJ4 {
            let ni = i.wrapping_add(di);
            let nj = j.wrapping_add(dj);
            if ni >= N || nj >= N {
                continue;
            }
            if used[ni][nj] {
                continue;
            }
            cand.push((ni, nj));
        }
        if cand.is_empty() {
            continue;
        }
        cand.shuffle(&mut rng);
        let (ni, nj) = cand[0];
        used[ni][nj] = true;
        C[ni][nj] = C[i][j];
        c += 1;
        cnt += 1;
    }

    for _ in 0..N * N - J - cnt {
        let mut visited = vec![vec![false; N]; N];
        let mut cnt: HashMap<usize, usize> = HashMap::new();
        let mut G: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut st: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for i in 0..N {
            for j in 0..N {
                if visited[i][j] {
                    continue;
                }
                let num = C[i][j];
                visited[i][j] = true;
                let mut c = 1;
                let mut Q = VecDeque::new();
                Q.push_back((i, j));
                st.entry(num).or_default().push((i, j));
                while let Some((pi, pj)) = Q.pop_front() {
                    for &(di, dj) in &DIJ4 {
                        let ni = pi.wrapping_add(di);
                        let nj = pj.wrapping_add(dj);
                        if ni >= N || nj >= N {
                            continue;
                        }
                        if num != C[ni][nj] {
                            G.entry(num).or_default().insert(C[ni][nj]);
                            continue;
                        }
                        if visited[ni][nj] {
                            continue;
                        }
                        visited[ni][nj] = true;
                        c += 1;
                        Q.push_back((ni, nj));
                        st.entry(num).or_default().push((ni, nj));
                    }
                }
                *cnt.entry(num).or_default() = c;
            }
        }

        let mut cand = vec![];
        for &i in cnt.keys() {
            for &next in G[&i].iter() {
                let mut u = i;
                let mut v = next;
                if u > v {
                    std::mem::swap(&mut u, &mut v);
                }
                let c = cnt[&u] + cnt[&v];
                cand.push((c, u, v));
            }
        }
        cand.sort();
        cand.dedup();
        let cmin = cand[0].0;
        let mut cand_min = vec![];
        for &(c, u, v) in cand.iter() {
            if c == cmin {
                cand_min.push((u, v));
            } else {
                break;
            }
        }
        let idx = rng.gen_range(0, cand_min.len());
        let (u, v) = cand_min[idx];
        for &(i, j) in st[&v].iter() {
            C[i][j] = u;
        }
    }

    let v = C.clone().into_iter().flatten().collect_vec();
    let mp = coordinate_compression(v);
    for i in 0..N {
        for j in 0..N {
            C[i][j] = mp[&C[i][j]] + 1;
        }
    }

    assert!(J == mp.len());

    for _ in 0..J - K {
        let mut G = vec![vec![]; J + 1];
        let mut outsides = vec![];
        let mut color_coords = vec![vec![]; J + 1];
        for i in 0..N {
            for j in 0..N {
                let c = C[i][j];
                color_coords[c].push((i, j));
                if c == 0 {
                    continue;
                }
                for &(di, dj) in &DIJ4 {
                    let ni = i.wrapping_add(di);
                    let nj = j.wrapping_add(dj);
                    if ni >= N || nj >= N {
                        outsides.push(c);
                        continue;
                    }
                    let nc = C[ni][nj];
                    if nc == 0 {
                        outsides.push(c);
                        continue;
                    }
                    if c == nc {
                        continue;
                    }
                    G[c].push(nc);
                }
            }
        }
        for i in 0..J {
            G[i].sort();
            G[i].dedup();
        }
        outsides.sort();
        outsides.dedup();
        outsides.shuffle(&mut rng);
        assert!(!outsides.is_empty());

        let mut low = LowLink::new(&G);
        low.build();
        let aps: BTreeSet<usize> = low.aps.into_iter().collect();
        for out in outsides.iter() {
            if !aps.contains(out) {
                for &(i, j) in color_coords[*out].iter() {
                    C[i][j] = 0;
                }
                break;
            }
        }
    }

    let v = C.clone().into_iter().flatten().collect_vec();
    let mp = coordinate_compression(v);
    for i in 0..N {
        for j in 0..N {
            C[i][j] = mp[&C[i][j]];
        }
    }

    // let mx = *C.iter().map(|v| v.iter().max().unwrap()).max().unwrap();
    // let mut doc = doc(600.0, 600.0);
    // let d = 600.0 / N as f32;
    // for i in 0..N {
    //     for j in 0..N {
    //         let color = if C[i][j] == 0 {
    //             "white".to_string()
    //         } else {
    //             color(C[i][j] as f64 / mx as f64)
    //         };
    //         let rec = rect(j as f32 * d, i as f32 * d, d, d, &color);
    //         let mut grp = group(format!("({}, {}): {}", i, j, C[i][j]));
    //         grp = grp.add(rec);
    //         doc = doc.add(grp);
    //     }
    // }
    // let vis = format!("<html><body>{}</body></html>", doc);
    // std::fs::write("gen.html", vis).unwrap();

    Input { N, K, L, AB, C }
}

fn coordinate_compression<T: std::cmp::Ord + Copy>(v: Vec<T>) -> BTreeMap<T, usize> {
    let mut vv = v;
    vv.sort();
    vv.dedup();
    let ret = vv.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    ret
}

#[derive(Debug, Clone)]
struct LowLink<'a> {
    size: usize,
    edge: &'a Vec<Vec<usize>>,
    visited: Vec<bool>,
    order: Vec<usize>,
    low: Vec<usize>,
    aps: Vec<usize>,
    bridge: Vec<(usize, usize)>,
}

impl<'a> LowLink<'a> {
    fn new(edge: &'a Vec<Vec<usize>>) -> Self {
        let size = edge.len();
        let visited = vec![false; size];
        let order = vec![0; size];
        let low = vec![0; size];
        let aps = vec![];
        let bridge = vec![];

        LowLink {
            size,
            edge,
            visited,
            order,
            low,
            aps,
            bridge,
        }
    }
    fn build(&mut self) {
        let mut cnt = 0;
        for i in 0..self.size {
            if !self.visited[i] {
                self.dfs(i, -1, &mut cnt);
            }
        }
        self.aps.sort();
        self.bridge.sort();
    }
    fn dfs(&mut self, pos: usize, parent: isize, cnt: &mut usize) {
        self.visited[pos] = true;
        self.order[pos] = *cnt;
        self.low[pos] = self.order[pos];
        *cnt += 1;
        let mut is_aps = false;
        let mut children_cnt = 0_usize;
        for &next in &self.edge[pos] {
            if !self.visited[next] {
                children_cnt += 1;
                self.dfs(next, pos as isize, cnt);
                self.low[pos] = self.low[pos].min(self.low[next]);
                if parent != -1 && self.order[pos] <= self.low[next] {
                    is_aps = true;
                }
                if self.order[pos] < self.low[next] {
                    self.bridge.push((pos.min(next), pos.max(next)));
                }
            } else if next as isize != parent {
                self.low[pos] = self.low[pos].min(self.order[next]);
            }
        }
        if parent == -1 && children_cnt >= 2 {
            is_aps = true;
        }
        if is_aps {
            self.aps.push(pos);
        }
    }
}
