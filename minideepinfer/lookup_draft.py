"""Propose continuations by matching a suffix in the committed token history."""


class LookupDraft:
    """Longest suffix first, latest earlier occurrence first; no match means no draft."""

    def __init__(self, max_match=4):
        if type(max_match) is not int or max_match < 1:
            raise ValueError('max_match must be a positive integer')
        self.max_match = max_match
        self.calls = self.hits = self.proposed = 0
        self.windows_checked = self.tokens_compared = 0
        self.last = None

    def __call__(self, prefix, limit):
        prefix = tuple(prefix)
        if not prefix or any(type(t) is not int or t < 0 for t in prefix):
            raise ValueError('prefix must contain nonnegative integer token IDs')
        if type(limit) is not int or limit < 0:
            raise ValueError('limit must be a nonnegative integer')
        self.calls += 1
        self.last = dict(context_length=len(prefix),limit=limit,match_length=0,source_start=None)
        if limit == 0:
            return ()
        for size in range(min(self.max_match,len(prefix)-1),0,-1):
            # Exclude the suffix itself; an earlier occurrence must have a continuation.
            for start in range(len(prefix)-size-1,-1,-1):
                self.windows_checked += 1
                matched = True
                for offset in range(size):
                    self.tokens_compared += 1
                    if prefix[start+offset] != prefix[len(prefix)-size+offset]:
                        matched = False
                        break
                if matched:
                    candidates = prefix[start+size:start+size+limit]
                    self.hits += 1
                    self.proposed += len(candidates)
                    self.last.update(match_length=size,source_start=start)
                    return candidates
        return ()

    def stats(self):
        return dict(calls=self.calls,matched_calls=self.hits,proposed_tokens=self.proposed,
                    windows_checked=self.windows_checked,tokens_compared=self.tokens_compared,
                    model_forward_calls=0,max_match=self.max_match)
