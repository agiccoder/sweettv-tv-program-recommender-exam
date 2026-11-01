"""PPR completion for globally cold items using aligned implicit-MF intensities."""
import math
from sweettv.models import _BaseModel
from sweettv.v3_models import native_scores

class ColdCompletion(_BaseModel):
    def __init__(self,base,factor,items,specification):
        super().__init__(base.data)
        self.base,self.factor,self.pool=base,factor,set(items)
        self.diagnostics.update({'method':specification['name'],'qualifying_family':'PPR/implicit-MF hybrid with explicit cold-item completion','formula':'Known items keep base PPR scores exactly. Cold score=MF_cold_score * sum_known(base_scores)/sum_known(MF_scores). Zero known MF mass gives no completion.','normalization':'Complete common pool before top50; no fallback padding, no per-item weight renormalization','future_viewing_used':False,'known_PPR_scores_unchanged':True})
    def native(self,user,candidates):
        if not set(candidates)<=self.pool:raise ValueError('Cannot expand frozen completion pool')
        base={i:float(s) for i,s in native_scores(self.base,user,self.pool).items() if math.isfinite(s) and s>0}
        mf={i:float(s) for i,s in native_scores(self.factor,user,self.pool).items() if math.isfinite(s) and s>0}
        denominator=math.fsum(s for i,s in mf.items() if i in self.data.item_index)
        scale=math.fsum(base.values())/denominator if denominator else 0.
        result={i:s for i,s in base.items() if i in candidates}
        if scale:
            for item,score in mf.items():
                if item not in self.data.item_index and item in candidates:result[item]=scale*score
        return result
