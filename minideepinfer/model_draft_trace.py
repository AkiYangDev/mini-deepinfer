"""Read-only host observations around the existing draft and verification calls."""


class DraftTrace:
    def __init__(self, draft, prompt):
        self.draft = draft
        self.report = dict(prompt_ids=list(prompt),capacity=draft.pool.capacity,initial_output=[],rounds=[])
        self.pending = None
        self.before = None

    def propose(self, prefix, limit):
        cache = self.draft.cache
        self.before = self.draft.stats()
        self.pending = dict(prefix_ids=list(prefix),limit=limit,
                            before_ids=list(cache.tokens) if cache else [],aligned_ids=None)
        values = self.draft(prefix,limit)
        cache = self.draft.cache
        self.pending.update(candidates=list(values),after_ids=list(cache.tokens) if cache else [])
        return values

    def aligned(self, cache):
        stats = self.draft.stats()
        added = stats['sync_input_tokens']-self.before['sync_input_tokens']
        self.pending.update(aligned_ids=list(cache.tokens),retained_before_sync=len(cache.tokens)-added,
                            sync_input_tokens=added,
                            replayed_input_tokens=stats['replayed_input_tokens']-self.before['replayed_input_tokens'],
                            recovered_branch=stats['recovered_branches']>self.before['recovered_branches'],
                            hybrid=cache.hybrid)

    def committed(self, tokens):
        if not self.report['initial_output']:
            self.report['initial_output'] = list(tokens)

    def observed(self, row):
        if list(row['draft']) != self.pending['candidates']:
            raise RuntimeError('observed candidates differ from the actual draft call')
        self.report['rounds'].append(dict(row,draft_state=self.pending))

    def closed(self):
        self.report['draft_cleanup'] = self.draft.stats()


def print_trace(report):
    """Print token IDs and logical lengths, not a device execution timeline."""
    print(f"\n{report['path']}，候选宽度 {report['width']}：")
    print(f"  prompt {report['prompt_ids']}；target 首次输出 {report['initial_output']}，末尾 token 待下一轮消费。")
    for number,row in enumerate(report['rounds'],1):
        state = row['draft_state']
        print(f"第 {number} 轮，已提交前缀 {state['prefix_ids']}")
        if state['limit']==0:
            print('  draft：候选上限为 0，本轮不计算。')
        else:
            restored = state['recovered_branch'] and state['hybrid']
            print(f"  draft 已算 {len(state['before_ids'])} → 保留 {state['retained_before_sync']} → 补到 {len(state['aligned_ids'])}；"
                  f"恢复线性状态={restored}，重算 {state['replayed_input_tokens']}，补算 {state['sync_input_tokens']} 个输入。")
            print(f"  提出 {state['candidates']}；draft 实际额外输入 {state['after_ids'][len(state['aligned_ids']):]}。")
        print(f"  target 输入 {row['input_ids']}，预测 {row['target']}。")
        role = dict(correction='target 改正',bonus='target 补 bonus',target='target 直接输出').get(row['final_role'],'候选中的 EOS')
        print(f"  接受 {row['accepted_draft']} 个候选，提交 {row['committed']}（{role}）。")
        ending = (f"最后输出 token={row['pending_token']}；请求结束，不再输入模型"
                  if row['finish_reason'] else f"末尾待计算 token={row['pending_token']}")
        print(f"  target KV 暂存 {row['tentative_cached']} → 保留 {row['retained_cached']}；"
              f"失效槽位 {row['invalidated_slots']}，恢复后重算 {row['replay_tokens']} 个输入；{ending}。")
    stats = report['draft_cleanup']
    print(f"结束：target 归还槽位 {report['target_free_slots']}/{report['capacity']}；draft {stats['free_slots']}/{report['capacity']}，"
          f"剩余线性状态 {stats['linear_states_remaining']}，保留快照={stats['snapshot_retained']}。")
