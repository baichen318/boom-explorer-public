# Author: baichen318@gmail.com

import abc
import os
from time import sleep
import threading
from typing import Callable, List
from multiprocessing.pool import ThreadPool
from utils import if_exist, execute, execute_with_subprocess, \
    timestamp, mkdir, remove, remove_suffix, info


def vlsi_timer(func):
    def wrapper(*args, **kwargs):
        # NOTICE: modify the thread name to specify the current running design
        threading.current_thread().name = args[2]
        start = timestamp()
        func(args[0], **kwargs)
        end = timestamp()
        msg = "total time: {} s".format(end - start)
        args[1].info(msg)
    return wrapper


def routine_error_handler(routine_check):
    def wrapper(*args, **kwargs):
        if routine_check(*args, **kwargs):
            pass
        else:
            error("routine is failed.")
    return wrapper


class VLSI(abc.ABC):
    def __init__(self):
        super(VLSI, self).__init__()

    @routine_error_handler
    def routine_check(
        self,
        period: int,
        cmd: str,
        condition: Callable,
        wait: int,
        *args: List,
        **kwargs: dict
    ) -> bool:
        """
            period: maximal seconds to execute func
            condition: condition check
            wait: waiting time in seconds
            args: parameters of `condition`
            kwargs: parameters of `condition`
        """
        start = timestamp()
        execute(cmd)
        while (timestamp() - start) < period:
            if condition(*args, **kwargs):
                return True
            sleep(wait)
        return False

    @routine_error_handler
    def routine_check_with_subprocess(
        self,
        period: int,
        cmd: str,
        condition: Callable,
        wait: int,
        *args: List,
        **kwargs: dict
    ):
        """
            period: maximal seconds to execute func
            condition: condition check
            wait: waiting time in seconds
        """
        start = timestamp()
        execute_with_subprocess(cmd)
        while (timestamp() - start) < period:
            if condition(*args, **kwargs):
                return True
            sleep(wait)
        return False

    @abc.abstractmethod
    def steps(self):
        """
            define VLSI steps
        """
        raise NotImplementedError()

    @vlsi_timer
    def run(self):
        for func in self.steps():
            func = getattr(self, func)
            func()


class Macros(abc.ABC):
    def __init__(self, configs: dict):
        self.macros = {}
        self.macros["chipyard-root"] = os.path.abspath(
            configs["vlsi-flow"]["chipyard-root"]
        )
        self.macros["workstation-root"] = os.path.join(
            self.macros["chipyard-root"],
            "workstation"
        )
        # NOTICE: this part relies on a self-modified version of
        # Chipyard, which is not public now.
        if_exist(self.macros["workstation-root"], strict=True)
        self.macros["generators-root"] = os.path.join(
            self.macros["chipyard-research-root"],
            "generators"
        )
        self.macros["tools-root"] = os.path.join(
            self.macros["chipyard-research-root"],
            "tools"
        )
        self.macros["project-root"] = os.path.join(
            self.macros["chipyard-research-root"],
            "project"
        )
        self.macros["sims-root"] = os.path.join(
            self.macros["chipyard-research-root"],
            "sims"
        )
        self.macros["vlsi-root"] = os.path.join(
            self.macros["chipyard-research-root"],
            "vlsi"
        )
        self.macros["customized-build-sbt"] = os.path.join(
            self.macros["workstation-root"],
            "misc",
            "build.sbt"
        )
        self.macros["default-ivy2-repo"] = os.path.join(
            self.macros["workstation-root"],
            "misc",
            "sbt-repo",
            ".ivy2"
        )
        self.macros["default-sbt-repo"] = os.path.join(
            self.macros["workstation-root"],
            "misc",
            "sbt-repo",
            ".sbt"
        )
        self.macros["default-coursier-repo"] = os.path.join(
            self.macros["workstation-root"],
            "misc",
            "sbt-repo",
            "coursier"
        )
        self.macros["sram-cache-json"] = self.get_sram_cache_json()
        self.macros["std-cells-lib-root"] = os.path.join(
            self.macros["vlsi-root"],
            "pdk",
            "asap7",
            "ASAP7_PDKandLIB_v1p5",
            "asap7libs_24.tar.bz2",
            "asap7libs_24",
            "lib"
        )
        self.macros["std-cells-db-root"] = os.path.join(
            self.macros["vlsi-root"],
            "pdk",
            "asap7",
            "ASAP7_PDKandLIB_v1p5",
            "asap7libs_24.tar.bz2",
            "asap7libs_24",
            "db"
        )
        self.macros["sram-macros-lib-root"] = os.path.join(
            self.macros["vlsi-root"],
            "pdk",
            "asap7",
            "sram",
            "lib"
        )
        self.macros["sram-macros-db-root"] = os.path.join(
            self.macros["vlsi-root"],
            "pdk",
            "asap7",
            "sram",
            "db",
        )

    def get_temp(self):
        return os.path.join(
            self.macros["chipyard-research-root"],
            "temp"
        )

    def get_generators(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            "generators"
        )

    def get_tools(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            "tools"
        )

    def get_project(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            "project"
        )

    def get_sims(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            "sims"
        )

    def get_ivy2_repo(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            ".ivy2"
        )

    def get_sbt_repo(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            ".sbt"
        )

    def get_coursier_repo(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            "coursier"
        )

    def get_sbt_repo_config(self):
        return os.path.join(
            self.get_temp(),
            self.soc,
            "repo.properties"
        )

    def get_sonatype(self):
        return os.path.join(
            self.get_coursier_repo(),
            "v1",
            "https",
            "oss.sonatype.org",
            "content",
            "repositories",
            "snapshots"
        )

    def get_maven(self):
        return os.path.join(
            self.get_coursier_repo(),
            "v1",
            "https",
            "repo1.maven.org",
            "maven2"
        )

    def get_eclipse(self):
        return os.path.join(
            self.get_coursier_repo(),
            "v1",
            "https",
            "download.eclipse.org",
            "jgit",
            "maven"
        )

    def get_sbt_plugin(self):
        return os.path.join(
            self.get_coursier_repo(),
            "v1",
            "https",
            "repo.scala-sbt.org",
            "scalasbt",
            "sbt-plugin-releases"
        )

    def get_artifactoryonline(self):
        return os.path.join(
            self.get_coursier_repo(),
            "v1",
            "https",
            "scalasbt.artifactoryonline.com",
            "scalasbt",
            "sbt-plugin-releases"
        )

    def get_simplytyped(self):
        return os.path.join(
            self.get_coursier_repo(),
            "v1",
            "https",
            "simplytyped.github.io",
            "repo",
            "releases"
        )

    def get_ivy2_cache(self):
        return os.path.join(
            self.get_ivy2_repo(),
            "cache"
        )

    def get_java_opts(self):
        return "-Xmx512G " \
            "-Xss128M " \
            "-XX:MaxPermSize=64G " \
            "-Djava.io.tmpdir={}/.java_tmp " \
            "-Dsbt.ivy.home={} " \
            "-Dsbt.global.base={} " \
            "-Dsbt.repository.config={} " \
            "-Dsbt.override.build.repos=true " \
            "-Dsbt.repository.secure=false " \
            "-Dsbt.coursier=false " \
            "-Dsbt.sourcemode=true " \
            "-Dsbt.workspace={}".format(
                self.macros["chipyard-research-root"],
                self.get_ivy2_repo(),
                self.get_sbt_repo(),
                self.get_sbt_repo_config(),
                self.get_tools()
            )

    def get_sram_cache_json(self):
        sram_json = os.path.join(
            self.macros["vlsi-root"],
            "hammer",
            "src",
            "hammer-vlsi",
            "technology",
            "asap7",
            "sram-cache.json"
        )
        if_exist(sram_json, strict=True)
        return sram_json

    def get_top_v(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "generated-src",
            "chipyard.TestHarness.{}".format(self.soc),
            "chipyard.TestHarness.{}.top.v".format(self.soc)
        )

    def get_top_mems_v(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "generated-src",
            "chipyard.TestHarness.{}".format(self.soc),
            "chipyard.TestHarness.{}.top.mems.v".format(self.soc)
        )

    def get_chiptop_mapped_v(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "syn-rundir",
            "ChipTop.mapped.v"
        )

    def get_simv(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            "simv"
        )

    def get_bmark_sim_root(self, bmark):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            bmark
        )

    def get_vpd(self, bmark):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            bmark,
            "{}.vpd".format(bmark)
        )

    def get_vcd(self, bmark):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            bmark,
            "{}.vcd".format(bmark)
        )

    def get_saif(self, bmark):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            bmark,
            "vcdplus.saif"
        )

    def get_dramsim2_ini(self):
        return os.path.join(
            self.macros["chipyard-research-root"],
            "generators",
            "testchipip",
            "src",
            "main",
            "resources",
            "dramsim2_ini"
        )

    def get_max_cycles(self):
        return 2000000

    def get_ucli_tcl(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            "run.tcl"
        )

    def get_force_regs(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            "force_regs.ucli"
        )

    def get_sim_report(self, bmark):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            bmark,
            "{}.out".format(bmark)
        )

    def get_sim_log(self, bmark):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "sim-syn-rundir",
            bmark,
            "{}.log".format(bmark)
        )

    def get_pt_root(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "hammer-synopsys-plugins",
            "power",
            "pt"
        )

    def get_search_path(self):
        return self.get_syn_root() + ' ' + \
            self.macros["std-cells-lib-root"] + ' ' + \
            self.macros["std-cells-db-root"] + ' ' + \
            self.macros["sram-macros-lib-root"] + ' ' + \
            self.macros["sram-macros-db-root"]

    def get_tech_library_files(self):
        tech_library_files = []
        for db in os.listdir(self.macros["std-cells-db-root"]):
            tech_library_files.append(
                os.path.join(
                    self.macros["std-cells-db-root"],
                    db
                )
            )
        for db in os.listdir(self.macros["sram-macros-db-root"]):
            tech_library_files.append(
                os.path.join(
                    self.macros["sram-macros-db-root"],
                    db
                )
            )
        return ' '.join(tech_library_files)

    def get_syn_root(self):
        return os.path.join(
            self.macros["vlsi-root"],
            "build",
            "chipyard.TestHarness.{}-ChipTop".format(self.soc),
            "syn-rundir"
        )


class VLSIFLow(VLSI, Macros):
    """
        VLSI FLow: push a single microarchitecture
        to the VLSI flow
    """
    def __init__(self, idx: int, vlsi_hammer_config: str, benchmarks: str):
        """
            idx: <int> the index of a microarchitecture
            vlsi_hammer_config: <str> the path to the VLSI HAMMER IR YAML configs.
            benchmarks: <list> the list of benchmarks
        """
        super(VLSIFLow, self).__init__()
        self._idx = idx
        self._vlsi_hammer_config = self.handle_vlsi_hammer_config(vlsi_hammer_config)
        self._benchmarks = self.handle_benchmarks(benchmarks)
        if isinstance(self.idx, int):
            self._soc = "Boom{}Config".format(self.idx)
        else:
            self._soc = self.idx
        self._build_root = os.path.join(self.get_temp(), self.soc)

    @property
    def idx(self):
        return self._idx

    @property
    def vlsi_hammer_config(self):
        return self._vlsi_hammer_config

    @property
    def benchmarks(self):
        return self._benchmarks

    @property
    def soc(self):
        return self._soc

    @property
    def build_root(self):
        return self._build_root

    def handle_vlsi_hammer_config(self, vlsi_hammer_config):
        vlsi_hammer_config = os.path.join(
            self.macros["vlsi-root"],
            "configs",
            "asap7",
            vlsi_hammer_config
        )
        if_exist(vlsi_hammer_config, strict=True)
        return vlsi_hammer_config

    def handle_benchmarks(self, benchmarks):
        _benchmarks = []
        for benchmark in benchmarks:
            _benchmarks.append(os.path.join(
                    self.macros["chipyard-research-root"],
                    "riscv-tools-install",
                    "riscv64-unknown-elf",
                    "share",
                    "riscv-tests",
                    "benchmarks",
                    benchmark + ".riscv"
                )
            )
            if_exist(_benchmarks[-1], strict=True)
        return _benchmarks

    def steps(self):
        steps = [
            "compile",
            "syn_to_sim",
            "sim",
            "power_analysis"
        ]
        return steps

    def compile(self):
        def duplicate_codes_repo():
            def generate_sbt_repo_config():
                repo_properties = "[repositories]\n" \
                    "  local\n" \
                    "  sonatype: file://{}\n" \
                    "  maven: file://{}\n" \
                    "  eclipse: file://{}\n" \
                    "  sbt-plugin: file://{}\n" \
                    "  artifactoryonline: file://{}\n" \
                    "  simplytyped: file://{}\n" \
                    "  cache: file://{}\n".format(
                        self.get_sonatype(),
                        self.get_maven(),
                        self.get_eclipse(),
                        self.get_sbt_plugin(),
                        self.get_artifactoryonline(),
                        self.get_simplytyped(),
                        self.get_ivy2_cache()
                    )
                return repo_properties

            mkdir(self.build_root)
            # generators
            execute(
                "rsync -a --exclude target {}/ {}".format(
                    self.macros["generators-root"],
                    self.get_generators()
                )
            )
            # tools
            execute(
                "rsync -a --exclude target {}/ {}".format(
                    self.macros["tools-root"],
                    self.get_tools()
                )
            )
            # project
            execute(
                "rsync -a --exclude project --exclude target {}/ {}".format(
                    self.macros["project-root"],
                    self.get_project()
                )
            )
            # sims
            execute(
                "rsync -a --exclude target {}/ {}".format(
                    self.macros["sims-root"],
                    self.get_sims()
                )
            )
            # build.sbt
            execute("cp -f {} {}".format(
                    self.macros["customized-build-sbt"],
                    os.path.join(self.build_root)
                )
            )
            # ivy2 repo.
            execute("cp -rf {} {}".format(
                    self.macros["default-ivy2-repo"],
                    self.get_ivy2_repo()
                )
            )
            # sbt repo.
            execute("cp -rf {} {}".format(
                    self.macros["default-sbt-repo"],
                    self.get_sbt_repo()
                )
            )
            # coursier repo.
            execute("cp -rf {} {}".format(
                    self.macros["default-coursier-repo"],
                    self.get_coursier_repo()
                )
            )
            # repo. config.
            with open(self.get_sbt_repo_config(), 'w') as f:
                f.writelines(generate_sbt_repo_config())
            info("{} is generated.".format(self.get_sbt_repo_config()))

        def condition():
            return if_exist(self.get_top_v()) and if_exist(self.get_top_mems_v())

        if condition():
            return
        duplicate_codes_repo()
        cmd = "cd {} && make build " \
            "SBT_BUILD_DIR={} " \
            "JAVA_OPTS=\"{}\" " \
            "INPUT_CONF=\"{}\" " \
            "CONFIG={} " \
            "MACRO_COMPILER=\"{}\"".format(
                self.macros["vlsi-root"],
                os.path.join(self.get_temp(), self.soc),
                self.get_java_opts(),
                self.vlsi_hammer_config,
                self.soc,
                self.macros["sram-cache-json"]
        )
        # time budget: 2 hours
        self.routine_check(2 * 3600, cmd, condition, 5 * 60)

    def synthesis(self):
        def remove_codes_repo():
            remove(self.build_root)

        def condition():
            return if_exist(self.get_chiptop_mapped_v())

        if condition():
            return
        cmd = "cd {} && make syn " \
            "SBT_BUILD_DIR={} " \
            "JAVA_OPTS=\"{}\" " \
            "INPUT_CONF=\"{}\" " \
            "CONFIG={}".format(
                self.macros["vlsi-root"],
                self.build_root,
                self.get_java_opts(),
                self.vlsi_hammer_config,
                self.soc
            )
        # time budget: 10h
        self.routine_check(10 * 3600, cmd, condition, 5 * 60)
        remove_codes_repo()

    def syn_to_sim(self):
        """
            from synthesis to constructing the simulator
        """
        def remove_codes_repo():
            remove(self.build_root)

        def condition():
            return if_exist(self.get_simv())

        if condition():
            return
        cmd = "cd {} && make sim-syn " \
            "SBT_BUILD_DIR={} " \
            "JAVA_OPTS=\"{}\" " \
            "INPUT_CONF=\"{}\" " \
            "CONFIG={}".format(
                self.macros["vlsi-root"],
                self.build_root,
                self.get_java_opts(),
                self.vlsi_hammer_config,
                self.soc
            )
        # time budget: 10.5h
        self.routine_check(10.5 * 3600, cmd, condition, 5 * 60)
        remove_codes_repo()

    def sim(self):
        def condition(bmark_alias):
            return if_exist(self.get_saif(bmark_alias))

        def generate_ucli_tcl():
            ucli_tcl = "source {}\nrun\nexit".format(self.get_force_regs())
            with open(self.get_ucli_tcl(), 'w') as f:
                f.writelines(ucli_tcl)
            info("{} is generated.".format(self.get_ucli_tcl()))

        generate_ucli_tcl()
        # NOTICE: we support parallel simulations
        p = ThreadPool(len(self.benchmarks))
        for bmark in self.benchmarks:
            bmark_alias = remove_suffix(os.path.basename(bmark), ".riscv")
            if condition(bmark_alias):
                continue
            mkdir(os.path.join(
                    self.macros["vlsi-root"],
                    "build",
                    "chipyard.TestHarness.{}-ChipTop".format(self.soc),
                    "sim-syn-rundir",
                    bmark_alias
                )
            )
            cmd = "set -o pipefail && cd {} && " \
                "{} " \
                "+permissive " \
                "+dramsim " \
                "+dramsim_ini_dir={} " \
                "+max-cycles={} " \
                "-ucli -do {} " \
                "+ntb_random_seed_automatic " \
                "+verbose " \
                "+vcdplusfile={} " \
                "+permissive-off " \
                "{} </dev/null " \
                "2> " \
                ">(spike-dasm > {}) | tee {}".format(
                    self.get_bmark_sim_root(bmark_alias),
                    self.get_simv(),
                    self.get_dramsim2_ini(),
                    self.get_max_cycles(),
                    self.get_ucli_tcl(),
                    self.get_vpd(bmark_alias),
                    bmark,
                    self.get_sim_report(bmark_alias),
                    self.get_sim_log(bmark_alias)
                )
            # time budget: 2h
            p.apply_async(
                self.routine_check_with_subprocess,
                (2 * 3600, cmd, condition, 5 * 60, bmark_alias,)
            )
        p.close()
        p.join()

    def power_analysis(self):
        def condition(bmark_alias):
            return if_exist(os.path.join(
                    self.get_bmark_sim_root(bmark_alias),
                    "power",
                    "reports",
                    "{}.power.avg.max.report".format(bmark_alias)
                )
            )

        # NOTICE: we support parallel power analysis
        p = ThreadPool(len(self.benchmarks))
        for bmark in self.benchmarks:
            bmark_alias = remove_suffix(os.path.basename(bmark), ".riscv")
            if condition(bmark_alias):
                continue
            execute("cp -f {} {}".format(
                    self.get_saif(bmark_alias),
                    os.path.join(
                        self.get_bmark_sim_root(bmark_alias),
                        bmark_alias + ".saif"
                    )
                )
            )
            # ptpx analysis
            cmd = "cd {} && make "\
                "build_pt_root=\"{}\" " \
                "vcs_vpd={} " \
                "search_path=\"{}\" " \
                "tech_library_files=\"{}\" " \
                "vcs_sim_root=\"{}\" " \
                "syn_root=\"{}\"".format(
                    self.get_pt_root(),
                    os.path.join(
                        self.get_bmark_sim_root(bmark_alias),
                        "power"
                    ),
                    bmark_alias,
                    self.get_search_path(),
                    self.get_tech_library_files(),
                    self.get_bmark_sim_root(bmark_alias),
                    self.get_syn_root()
            )
            # time budget: 1h
            p.apply_async(
                self.routine_check,
                (1 * 3600, cmd, condition, 5 * 60, bmark_alias,)
            )
        p.close()
        p.join()


def construct_vlsi_manager(
    idx: int, vlsi_hammer_config: str, benchmarks: str
) -> VLSIFLow:
    vlsi_manager = VLSIFLow(idx, vlsi_hammer_config, benchmarks)
    return vlsi_manager 
