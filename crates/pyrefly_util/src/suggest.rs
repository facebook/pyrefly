/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;

/// The largest edit distance worth suggesting. Every tier is bounded by it, so
/// a pair whose lengths differ by more than this cannot match whatever tier they
/// fall in, which lets a caller dismiss a candidate before it is built.
///
/// Bounding it is also what keeps the search cheap: the band the distance
/// computation has to fill is at most `2 * MAX_DISTANCE + 1` cells wide, so it
/// fits on the stack.
const MAX_DISTANCE: usize = 3;

/// How far apart a candidate may be from the missing name and still be worth
/// suggesting. Longer names tolerate more typos.
///
/// This scales with the missing name alone. How many typos are plausible is a
/// property of what the author typed, not of whatever happens to be in scope,
/// and keeping the candidate out of it makes the bound a constant for the whole
/// search rather than something to recompute per pair.
fn max_distance_for(missing_len: usize) -> usize {
    match missing_len {
        0..=4 => 1,
        5..=8 => 2,
        _ => MAX_DISTANCE,
    }
}

/// Edit distance between `a` and `b`, saturating at `bound + 1`.
///
/// Callers only ever care whether a candidate is *close*, so computing an exact
/// distance for one that is far away is wasted work. Three bounds make that
/// cheap. Two names whose lengths differ by more than `bound` cannot be within
/// it, because every edit changes the length by at most one; a path costing at
/// most `bound` cannot stray more than `bound` cells from the diagonal, so only
/// that band needs filling; and once every cell in a row exceeds `bound`, no
/// continuation can come back under it, so the rest of the table is abandoned.
///
/// Credit to Mistral Contrastin, who found that the same "Did you mean?" search
/// dominated the Hack typechecker on files that had no errors at all, and fixed
/// it there by giving the distance function an upper bound.
///
/// `bound` comes from [`max_distance_for`], so it never exceeds
/// [`MAX_DISTANCE`] and the two band rows can be fixed-size arrays.
fn distance_at_most<T: PartialEq>(a: &[T], b: &[T], bound: usize) -> usize {
    debug_assert!(bound <= MAX_DISTANCE, "band is sized for MAX_DISTANCE");
    let (n, m) = (a.len(), b.len());
    if n.abs_diff(m) > bound {
        return bound + 1;
    }
    let over = bound + 1;
    // Cell `(i, j)` of the table lives at band offset `j + bound - i`, so a row
    // needs `2 * bound + 1` slots plus one more for the `(i - 1, j)` neighbour.
    let width = 2 * bound + 2;
    let mut prev = [over; 2 * MAX_DISTANCE + 2];
    let mut curr = [over; 2 * MAX_DISTANCE + 2];
    // Row 0: matching an empty prefix of `a` costs one deletion per character.
    for j in 0..=bound.min(m) {
        prev[j + bound] = j;
    }
    for i in 1..=n {
        // Clearing the row is cheaper than tracking which slots the shifting
        // band leaves stale: it is a couple of stores on a fixed-size array, and
        // skipping it measured slower.
        //
        // The cells stay `usize` even though every one of them fits in a `u8`.
        // The band is two cache lines at most, so there is no memory traffic to
        // save, while narrower cells have to be widened on every load and
        // truncated on every store to mix with the `usize` offsets — `u32`
        // measured 18% slower here and `u8` 29% slower.
        curr[..width].fill(over);
        let a_i = &a[i - 1];
        let mut row_min = over;
        // Column 0 is only inside the band for the first `bound` rows.
        if i <= bound {
            curr[bound - i] = i;
            row_min = i;
        }
        for j in i.saturating_sub(bound).max(1)..=(i + bound).min(m) {
            // `j + bound - i`, not `j - i + bound`: `j` can be below `i` and
            // these are unsigned.
            let o = j + bound - i;
            let substitute = prev[o] + usize::from(*a_i != b[j - 1]);
            let delete = prev[o + 1] + 1;
            // `o == 0` means the cell to the left fell outside the band, which
            // can only mean it already cost more than `bound`.
            let insert = if o > 0 { curr[o - 1] + 1 } else { over };
            let cell = substitute.min(delete).min(insert).min(over);
            curr[o] = cell;
            row_min = row_min.min(cell);
        }
        if row_min > bound {
            return over;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m + bound - n].min(over)
}

/// A name the search may offer, and what it needs to know about it.
pub struct Candidate<'a> {
    name: &'a Name,
    /// The name's length in characters.
    ///
    /// Nearly always the only thing the search looks at, since a candidate too
    /// far from the missing name in length cannot be within the edit distance.
    char_len: usize,
    /// Smaller wins a tie on distance, so callers use it to prefer nearer
    /// scopes.
    priority: usize,
}

impl<'a> Candidate<'a> {
    /// A candidate whose length the caller has already recorded somewhere cheap
    /// to reach -- which one scanning a large scope should, rather than have it
    /// counted here once per lookup.
    pub fn new(name: &'a Name, char_len: usize, priority: usize) -> Self {
        Self {
            name,
            char_len,
            priority,
        }
    }

    /// A candidate whose length is not already known, measured on the spot.
    pub fn measured(name: &'a Name, priority: usize) -> Self {
        Self::new(name, char_len(name), priority)
    }
}

/// The state a search carries between candidates.
///
/// Kept between candidates so that a match found early narrows what everything
/// after it is measured against.
pub struct Search<'a> {
    missing: &'a str,
    /// For ASCII a byte is a character, which is what lets the common path
    /// compare bytes and skip decoding entirely.
    missing_ascii: bool,
    missing_len: usize,
    /// Built on demand: only the character path needs it, and that path is
    /// reached only when one of the two names is not ASCII.
    missing_chars: Option<Vec<char>>,
    candidate_chars: Vec<char>,
    best: Option<(Name, usize, usize)>,
    /// The widest distance still worth considering: the distance of the best
    /// match so far, or what the missing name's length allows before there is
    /// one. Only ever decreases, so it is also the bound every comparison is
    /// made against.
    best_distance: usize,
}

impl<'a> Search<'a> {
    pub fn new(missing: &'a Name) -> Self {
        let missing = missing.as_str();
        let missing_ascii = missing.is_ascii();
        let missing_len = if missing_ascii {
            missing.len()
        } else {
            missing.chars().count()
        };
        Self {
            missing,
            missing_ascii,
            missing_len,
            missing_chars: None,
            candidate_chars: Vec::new(),
            best: None,
            best_distance: max_distance_for(missing_len),
        }
    }

    /// Whether length alone rules a candidate out, against the running bound.
    ///
    /// A caller that holds a candidate's length can ask before assembling it.
    /// Every rejection here is a distance computation not started.
    #[inline]
    fn rejects(&self, candidate: &Candidate) -> bool {
        // A single letter is noise as a suggestion whatever its distance.
        candidate.char_len == 1
            // The distance is never smaller than the difference in length, so a
            // candidate further away than the bound cannot come within it.
            || self.missing_len.abs_diff(candidate.char_len) > self.best_distance
    }

    /// Measure a candidate [`Search::rejects`] could not dismiss, keeping it if
    /// it is the nearest so far.
    #[inline(never)]
    fn consider(&mut self, candidate: Candidate) {
        let Candidate {
            name,
            char_len: candidate_len,
            priority,
        } = candidate;
        let candidate_str = name.as_str();
        let byte_len = candidate_str.len();
        // `best_distance` starts at what the missing name's length allows and
        // only narrows, so it is the bound this comparison is made against --
        // there is nothing further to work out, and `rejects` has already
        // established the two are close enough in length to be worth comparing.
        let bound = self.best_distance;
        // Matching counts mean every character was one byte, so the two are
        // pure ASCII and can be compared as bytes -- no characters to
        // materialise, and the answer is identical.
        let distance = if self.missing_ascii && candidate_len == byte_len {
            distance_at_most(self.missing.as_bytes(), candidate_str.as_bytes(), bound)
        } else {
            let missing = self.missing;
            let missing_chars = self
                .missing_chars
                .get_or_insert_with(|| missing.chars().collect());
            self.candidate_chars.clear();
            self.candidate_chars.extend(candidate_str.chars());
            distance_at_most(missing_chars, &self.candidate_chars, bound)
        };
        // `distance_at_most` saturates at `bound + 1`, so anything over the
        // bound is a candidate that could not come close enough.
        if distance == 0 || distance > bound {
            return;
        }
        match &self.best {
            Some((_, best_distance, best_priority))
                if distance > *best_distance
                    || (distance == *best_distance && priority >= *best_priority) => {}
            _ => {
                self.best = Some((name.clone(), distance, priority));
                self.best_distance = distance;
            }
        }
    }

    /// Offer a candidate, if `admit` allows it.
    ///
    /// `admit` is consulted only for a candidate the bound did not already
    /// dismiss, so a caller with an expensive test -- a lookup that says
    /// whether the name is really in scope, say -- can put it here and not pay
    /// for it on the candidates that were never in the running.
    #[inline]
    pub fn offer_if(&mut self, candidate: Candidate, admit: impl FnOnce() -> bool) {
        if !self.rejects(&candidate) && admit() {
            self.consider(candidate);
        }
    }

    /// Offer a candidate with nothing further to check.
    #[inline]
    pub fn offer(&mut self, candidate: Candidate) {
        self.offer_if(candidate, || true);
    }

    /// The nearest candidate the search kept, if any.
    pub fn finish(self) -> Option<Name> {
        self.best.map(|(name, _, _)| name)
    }
}

/// A name's length in characters. Cheap for ASCII, which identifiers almost
/// always are.
#[inline]
fn char_len(name: &Name) -> usize {
    let s = name.as_str();
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// Pick the closest candidate to `missing`, preferring smaller `priority` on ties.
///
/// Each candidate arrives with its length in characters, because the caller
/// generating them has already worked it out to filter on.
///
/// A convenience for callers that already hold their candidates. One that
/// generates them should drive a [`Search`] instead, so that a candidate the
/// bound has already ruled out is never built.
pub fn best_suggestion<'a, I>(missing: &Name, candidates: I) -> Option<Name>
where
    I: IntoIterator<Item = Candidate<'a>>,
{
    let mut search = Search::new(missing);
    for candidate in candidates {
        search.offer(candidate);
    }
    search.finish()
}

#[cfg(test)]
mod tests {
    use strsim::levenshtein;

    use super::*;

    fn suggest(missing: &str, candidates: &[&str]) -> Option<String> {
        let names: Vec<Name> = candidates.iter().map(Name::new).collect();
        best_suggestion(
            &Name::new(missing),
            names
                .iter()
                .enumerate()
                .map(|(i, n)| Candidate::measured(n, i)),
        )
        .map(|n| n.as_str().to_owned())
    }

    /// The band arrays are sized for `MAX_DISTANCE`, so `max_distance_for`
    /// exceeding it would overflow them. Pin the ceiling rather than leave it
    /// to the doc comment.
    #[test]
    fn max_distance_for_never_exceeds_the_published_ceiling() {
        for longest in 0..512 {
            assert!(
                max_distance_for(longest) <= MAX_DISTANCE,
                "max_distance_for({longest}) exceeded MAX_DISTANCE"
            );
        }
    }

    #[test]
    fn candidate_at_the_length_limit_is_still_found() {
        // The bound comes from the missing name alone: `columns` is seven
        // characters, which allows two edits however long the candidate is. Two
        // characters longer sits on the boundary and must survive.
        assert_eq!(
            suggest("columns", &["columnsxy"]),
            Some("columnsxy".to_owned())
        );
        // Three characters longer needs at least three edits, so it is out --
        // the candidate's own length does not buy it a wider allowance.
        assert_eq!(suggest("columns", &["columnsxyz"]), None);
    }

    /// The tier `max_distance_for` picks has to be chosen on the same units the
    /// distance is measured in. A non-ASCII name has more bytes than characters,
    /// so measuring the tier in bytes would quietly allow a further candidate.
    #[test]
    fn distance_tiers_are_measured_in_characters() {
        // Four characters, eight bytes. Four characters allow one edit; eight
        // bytes would allow two, so a candidate two edits away must not win.
        assert_eq!(
            suggest("\u{3b1}\u{3b2}\u{3b3}\u{3b4}", &["\u{3b1}\u{3b2}xy"]),
            None
        );
        // One edit away is inside the tier either way.
        assert_eq!(
            suggest("\u{3b1}\u{3b2}\u{3b3}\u{3b4}", &["\u{3b1}\u{3b2}\u{3b3}y"]),
            Some("\u{3b1}\u{3b2}\u{3b3}y".to_owned())
        );
    }

    #[test]
    fn non_ascii_names_are_compared_by_character() {
        // `α` is one character but two bytes, so anything working in bytes
        // would both mis-measure the length and mis-count the substitution.
        assert_eq!(suggest("xyz", &["xy\u{3b1}"]), Some("xy\u{3b1}".to_owned()));
        assert_eq!(
            suggest("\u{3b1}\u{3b2}\u{3b3}", &["\u{3b1}\u{3b2}\u{3b4}"]),
            Some("\u{3b1}\u{3b2}\u{3b4}".to_owned())
        );
        // A non-ASCII candidate against an ASCII missing name, and the reverse.
        assert_eq!(suggest("abc", &["ab\u{3b1}"]), Some("ab\u{3b1}".to_owned()));
        assert_eq!(suggest("ab\u{3b1}", &["abc"]), Some("abc".to_owned()));
    }

    #[test]
    fn a_closer_candidate_still_wins_after_a_farther_one() {
        // The running best tightens the bound for later candidates, so this
        // pins that tightening it does not hide a better match.
        assert_eq!(
            suggest("column", &["columnabc", "columns", "col"]),
            Some("columns".to_owned())
        );
        // Earliest wins on a tie, which is what `priority` encodes.
        assert_eq!(
            suggest("column", &["columns", "columnz"]),
            Some("columns".to_owned())
        );
    }

    /// The band, the row-abandonment and the length check are all easy to get
    /// subtly wrong, so check every one of them against a reference
    /// implementation rather than against hand-written expectations.
    #[test]
    fn bounded_distance_agrees_with_the_unbounded_one_below_the_bound() {
        let alphabet = ["a", "b", "\u{3b1}"];
        let mut corpus: Vec<String> = vec![String::new()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for base in &corpus {
                for letter in alphabet {
                    next.push(format!("{base}{letter}"));
                }
            }
            corpus.extend(next);
        }
        corpus.push("kitten".to_owned());
        corpus.push("sitting".to_owned());
        corpus.push("ColumnSchema".to_owned());
        corpus.push("IdListColumnType".to_owned());

        for a in &corpus {
            for b in &corpus {
                let exact = levenshtein(a, b);
                let a_chars: Vec<char> = a.chars().collect();
                let b_chars: Vec<char> = b.chars().collect();
                for bound in 0..=MAX_DISTANCE {
                    let got = distance_at_most(&a_chars, &b_chars, bound);
                    let want = if exact <= bound { exact } else { bound + 1 };
                    assert_eq!(got, want, "{a:?} vs {b:?} at bound {bound}");
                    // ASCII pairs also go through the byte path, which has to
                    // agree cell for cell with the character one.
                    if a.is_ascii() && b.is_ascii() {
                        let bytes = distance_at_most(a.as_bytes(), b.as_bytes(), bound);
                        assert_eq!(bytes, want, "bytes {a:?} vs {b:?} at bound {bound}");
                    }
                }
            }
        }
    }
}
