# Author: baichen318@gmail.com


import os
import numpy as np
from typing import List, NoReturn
from collections import OrderedDict
from .design_space import DesignSpace, Macros
from utils import info, if_exist, mkdir, remove, load_excel


class BOOMMacros(Macros):
    def __init__(self, configs):
        super(BOOMMacros, self).__init__(configs)
        self.macros["core-cfg"] = os.path.join(
            self.macros["chipyard-root"],
            "generators",
            "boom",
            "src",
            "main",
            "scala",
            "common",
            "config-mixins.scala"
        )
        self.macros["soc-cfg"] = os.path.join(
            self.macros["chipyard-root"],
        "generators",
            "chipyard",
            "src",
            "main",
            "scala",
            "config",
            "BoomConfigs.scala"
        )
        self.validate_macros()

    def validate_macros(self):
        if_exist(self.macros["core-cfg"], strict=True)
        if_exist(self.macros["soc-cfg"], strict=True)

    def get_mapping_params(self, vec, idx):
        return self.components_mappings[self.components[idx]][vec[idx]]

    def get_vec_params(self, elem_of_microarchitecture_embedding, idx):
        for k, v in self.components_mappings[self.components[idx]].items():
            if v[0] == elem_of_microarchitecture_embedding:
                return k

    def generate_branch_predictor(self) -> str:
        """
            default branch predictor: TAGEL
        """
        return "new WithTAGELBPD ++"

    def generate_fetch_width(self, vec: List[int]) -> int:
        return vec[0]

    def generate_decode_width(self, vec: List[int]) -> int:
        return vec[7]

    def generate_fetch_buffer_entries(self, vec: List[int]) -> int:
        return vec[1]

    def generate_rob_entries(self, vec: List[int]) -> int:
        return vec[8]

    def generate_ras_entries(self, vec: List[int]) -> int:
        return vec[2]

    def generate_phy_registers(self, vec: List[int]) -> str:
        return """numIntPhysRegisters = %d,
                    numFpPhysRegisters = %d""" % (
                vec[9], vec[10]
            )

    def generate_lsu(self, vec: List[int]) -> str:
        return """numLdqEntries = %d,
                    numStqEntries = %d""" % (
                vec[14], vec[15]
            )

    def generate_max_br_count(self, vec: List[int]) -> int:
        return vec[3]

    def generate_issue_parames(self, vec: List[int]) -> int:
        isu_params = [
            # IQT_MEM.numEntries IQT_MEM.dispatchWidth
            # IQT_INT.numEntries IQT_INT.dispatchWidth
            # IQT_FP.numEntries IQT_FP.dispatchWidth
            [8, vec[7], 8, vec[7], 8, vec[7]],
            [12, vec[7], 20, vec[7], 16, vec[7]],
            [16, vec[7], 32, vec[7], 24, vec[7]],
            [24, vec[7], 40, vec[7], 32, vec[7]],
            [24, vec[7], 40, vec[7], 32, vec[7]]
        ]
        # select specific BOOM
        _isu_params = isu_params[vec[7] - 1]
        return """Seq(
                        IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_MEM.litValue, dispatchWidth=%d),
                        IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_INT.litValue, dispatchWidth=%d),
                        IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_FP.litValue, dispatchWidth=%d)
                    )""" % (
                vec[11], _isu_params[0], _isu_params[1],
                vec[12], _isu_params[2], _isu_params[3],
                vec[13], _isu_params[4], _isu_params[5]
            )

    def generate_ftq_entries(self, vec):
        ftq_entries = [8, 16, 24, 32, 32]
        return ftq_entries[vec[7]]

    def generate_dcache_and_mmu(self, vec):
        return """Some(
                    DCacheParams(
                        rowBits=site(SystemBusKey).beatBits,
                        nSets=64,
                        nWays=%d,
                        nMSHRs=%d,
                        nTLBSets=1,
                        nTLBWays=%d
                    )
                    )""" % (
              vec[16],
              vec[17],
              vec[18]
            )

    def generate_icache_and_mmu(self, vec):
        return """Some(
                      ICacheParams(
                        rowBits=site(SystemBusKey).beatBits,
                        nSets=64,
                        nWays=%d,
                        nTLBSets=1,
                        nTLBWays=%d,
                        fetchBytes=%d*4
                      )
                    )""" % (
                vec[4],
                vec[5],
                vec[6]
            )

    def generate_system_bus_key(self, vec):
        return vec[0] << 1

    def generate_core_cfg_impl(self, name: str, vec: List[int]) -> str:
        codes = '''
class %s(n: Int = 1, overrideIdOffset: Option[Int] = None) extends Config(
  %s
  new Config((site, here, up) => {
    case TilesLocated(InSubsystem) => {
      val prev = up(TilesLocated(InSubsystem), site)
      val idOffset = overrideIdOffset.getOrElse(prev.size)
      (0 until n).map { i =>
        BoomTileAttachParams(
          tileParams = BoomTileParams(
            core = BoomCoreParams(
              fetchWidth = %d,
              decodeWidth = %d,
              numFetchBufferEntries = %d,
              numRobEntries = %d,
              numRasEntries = %d,
              %s,
              %s,
              maxBrCount = %d,
              issueParams = %s,
              ftq = FtqParameters(nEntries=%d),
              fpu = Some(
                freechips.rocketchip.tile.FPUParams(
                  sfmaLatency=4, dfmaLatency=4, divSqrt=true
                )
              ),
              enablePrefetching = true
            ),
            dcache = %s,
            icache = %s,
            hartId = i + idOffset
          ),
          crossingParams = RocketCrossingParams()
        )
      } ++ prev
    }
    case SystemBusKey => up(SystemBusKey, site).copy(beatBytes = %d)
    case XLen => 64
  })
)
''' % (
    name,
    self.generate_branch_predictor(),
    self.generate_fetch_width(vec),
    self.generate_decode_width(vec),
    self.generate_fetch_buffer_entries(vec),
    self.generate_rob_entries(vec),
    self.generate_ras_entries(vec),
    self.generate_phy_registers(vec),
    self.generate_lsu(vec),
    self.generate_max_br_count(vec),
    self.generate_issue_parames(vec),
    self.generate_ftq_entries(vec),
    self.generate_dcache_and_mmu(vec),
    self.generate_icache_and_mmu(vec),
    self.generate_system_bus_key(vec)
)
        return codes

    def write_core_cfg_impl(self, codes: str) -> NoReturn:
        with open(self.macros["core-cfg"], 'a') as f:
            f.writelines(codes)

    def generate_soc_cfg_impl(self, soc_name: str, core_name: str) -> NoReturn:
        codes = '''
class %s extends Config(
  new boom.common.%s(1) ++
  new chipyard.config.AbstractConfig)
''' % (
        soc_name,
        core_name
    )
        return codes

    def write_soc_cfg_impl(self, codes: str) -> NoReturn:
        with open(self.macros["soc-cfg"], 'a') as f:
            f.writelines(codes)


class BOOMDesignSpace(DesignSpace, BOOMMacros):
    def __init__(self, configs: dict, design_space: dict):
        """
        example:
            design_space: {
                "FetchWidth": [4, 8],
                ...
            }
        """
        self.configs = configs
        self.design_space = design_space
        size, self.component_dims = self.construct_design_space_size()
        DesignSpace.__init__(self, size, len(self.design_space.keys()))
        BOOMMacros.__init__(self, self.configs)

    def construct_design_space_size(self):
        s = []
        for k, v in self.design_space.items():
            s.append(len(k))
        return np.prod(s), s

    def idx_to_vec(self, idx: int) -> List[int]:
        idx -= 1
        assert idx >= 0, \
            assert_error("invalid index.")
        assert idx < self.size, \
            assert_error("index exceeds the search space.")
        vec = []
        for dim in self.component_dims:
            vec.append(idx % dim)
            idx //= dim
        return vec

    def vec_to_idx(self, vec):
        idx = 0
        for j, k in enumerate(vec):
            idx += int(np.prod(self.component_dims[:j])) * k
        assert idx >= 0, \
            assert_error("invalid index.")
        assert idx < self.size, \
            assert_error("index exceeds the search space.")
        idx += 1
        return idx

    def generate_core_cfg(self, batch: int) -> str:
        """
            generate core configurations
        """
        codes = []
        for idx in batch:
            codes.append(self.generate_core_cfg_impl(
                    "WithN{}Booms".format(idx),
                    self.idx_to_vec(idx)
                )
            )
        return codes

    def write_core_cfg(self, codes: str) -> NoReturn:
        self.write_core_cfg_impl(codes)

    def generate_soc_cfg(self, batch: int) -> NoReturn:
        """
            generate soc configurations
        """
        codes = []
        for idx in batch:
            codes.append(self.generate_soc_cfg_impl(
                    "Boom{}Config".format(idx),
                    "WithN{}Booms".format(idx)
                )
            )
        return codes

    def generate_chisel_codes(self, batch: List[int]) -> NoReturn:
        codes = self.generate_core_cfg(batch)
        self.write_core_cfg(codes)
        codes = self.generate_soc_cfg(batch)
        self.write_soc_cfg(codes)

    def write_soc_cfg(self, codes):
        self.write_soc_cfg_impl(codes)


def parse_boom_design_space(configs: dict) -> BOOMDesignSpace:
    sheet = load_excel(configs["design-space"], sheet_name="BOOM Design Space")
    design_space = OrderedDict()
    for row in sheet.values:
        design_space[row[1]] = []
        for val in row[-1].split(','):
            design_space[row[1]].append(int(val))
    return BOOMDesignSpace(
        configs,
        design_space
    )
