
import pandas as pd
import pytest
import subprocess


@pytest.fixture(scope="session")
def minc_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("files")
    paths = [str(d / f"img_{i}.mnc") for i in range(3)]
    for p in paths:
        subprocess.check_call(['rawtominc', p, "-osigned", "-ofloat", "-input", "/dev/urandom", "100", "120", "140"])
    return paths

@pytest.fixture(scope="session")
def files(minc_files):
    return ["--files"] + minc_files

@pytest.fixture(scope="session")
def csv_file(tmp_path_factory, minc_files):
    csv_file = tmp_path_factory.mktemp("csv") / "files.csv"
    df = pd.DataFrame({ 'file' : minc_files })
    df.to_csv(csv_file, index=False)
    return [f"--csv-file={csv_file}"]

@pytest.fixture(scope="session")
def initial_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("initial_model")
    standard = str(d / "model.mnc")
    standard_mask = str(d / "model_mask.mnc")
    native = str(d / "model_native.mnc")
    native_mask = str(d / "model_native_mask.mnc")
    for p in [standard, standard_mask, native, native_mask]:
        subprocess.check_call(['rawtominc', p, "-osigned", "-ofloat", "-input", "/dev/urandom", "100", "120", "140"])
    transform = str(d / "model_native_to_standard.xfm")
    subprocess.check_call(['param2xfm', transform])
    return standard

chain_time_points = [23, 35, 65]

@pytest.fixture(scope="session")
def pride_of_models(tmp_path_factory):
    d = tmp_path_factory.mktemp("pride_of_models")
    df = pd.DataFrame([{ 'model_file' : str(d / f"img_{t}.mnc"),
                         'time_point' : t}
                       for t in chain_time_points])
    for p in list(df.model_file) + [str(d / f"img_{t}_mask.mnc") for t in chain_time_points]:
        subprocess.check_call(['rawtominc', p, "-osigned", "-ofloat", "-input", "/dev/urandom", "100", "120", "140"])
    out = tmp_path_factory.mktemp("csv") / "pride_of_models.csv"
    df.to_csv(out, index=False)
    return out


@pytest.fixture(scope="session")
def chain_csv_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("files")
    n_subjects = 3
    df = pd.DataFrame([{ 'filename': str(d / f"img_t{t}_id{n}.mnc"),
                         'timepoint': t,
                         'subject_id' : n }
                       for n in range(n_subjects) for t in chain_time_points])
    for p in df.filename:
        subprocess.check_call(['rawtominc', p, "-osigned", "-ofloat", "-input", "/dev/urandom", "100", "120", "140"])
    out = tmp_path_factory.mktemp("csv") / "subjects.csv"
    df.to_csv(out, index=False)
    return out


@pytest.fixture(scope="session")
def atlas_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("atlas")
    n_atlases = 3
    anatomical_images = [str(d / f"atlas_{i}.mnc") for i in range(n_atlases)]
    label_images = [str(d / f"atlas_{i}_labels.mnc") for i in range(n_atlases)]
    mask_images = [str(d / f"atlas_{i}_mask.mnc") for i in range(n_atlases)]
    for p in anatomical_images + label_images + mask_images:
        subprocess.check_call(['rawtominc', p, "-osigned", "-ofloat", "-input", "/dev/urandom", "100", "120", "140"])
    return str(d)


pipelines = ["asymmetry.py",
             "MBM.py",
             "twolevel_model_building.py",
             "registration_chain.py",
             "NLIN.py",
             "LSQ6.py",
             "LSQ12.py"]


class TestTrivialUsage:
    def test_help(self, script_runner):
        for prog in pipelines:
            ret = script_runner.run(prog, "--help")
            assert ret.success

    def test_version(self, script_runner):
        for prog in pipelines:
            ret = script_runner.run(prog, '--version')
            assert ret.success


general_args = ["--pipeline-name=test-pipe",
                "--no-execute",
                #"--no-check-input-files",
                "--no-check-commands-exist"]

n3_args = [["--nuc"], ["--no-nuc"], []]

inorm_args = [["--inorm"], ["--no-inorm"], []]

@pytest.fixture(scope="session")
def lsq6_targets(initial_model, minc_files):
    return ["--bootstrap", f"--lsq6-target={minc_files[0]}", f"--init-model={initial_model}"]

lsq6_methods = ["--lsq6-large-rotations", "--lsq6-centre-estimation", "--lsq6-simple"]

@pytest.fixture(scope="session")
def lsq12_protocol(tmp_path_factory):
    protocol_file = tmp_path_factory.mktemp("linear") / "lsq12_protocol.csv"
    protocol = ( 
      '"blur";0.28;0.19;0.14\n'
      '"step";0.9;0.46;0.3\n'
      '"gradient";FALSE;TRUE;FALSE\n'
    )
    with open(protocol_file, 'w') as f:
        f.write(protocol)
    return protocol_file

@pytest.fixture(scope="session")
def nonlinear_protocol(tmp_path_factory):
    protocol_file = tmp_path_factory.mktemp("nonlinear") / "MAGeT_protocol.csv"
    protocol = (
      '"blur";0.25;0.25;0.25;0.25;0.25;-1\n'
      '"step";1;0.5;0.5;0.2;0.2;0.1\n'
      '"iterations";60;60;60;10;10;4\n'
      '"simplex";3;3;3;1.5;1.5;1\n'
      '"gradient";False;False;True;False;True;False\n'
      '"w_translations";0.2;0.2;0.2;0.2;0.2;0.2\n'
      '"optimization";"-use_simplex";"-use_simplex";"-use_simplex";"-use_simplex";"-use_simplex";"-use_simplex"\n'
    )
    with open(protocol_file, 'w') as f:
        f.write(protocol)
    return protocol_file

registration_methods = [["--registration-method=ANTS"],
                        #["--registration-method=minctracc"],
                        []]

registration_strategies = [[]] + [[f"--registration-strategy={s}"] for s in ["build_model", "pairwise", "pairwise_and_build_model"]]


class TestLSQ6:
    @pytest.mark.parametrize('n3_arg', n3_args)
    @pytest.mark.parametrize('inorm_arg', inorm_args)
    @pytest.mark.parametrize('lsq6_method', lsq6_methods)
    @pytest.mark.parametrize('use_csv', (True, False))
    def test(self, script_runner, files, csv_file, use_csv, initial_model, lsq6_method, n3_arg, inorm_arg, lsq6_targets):
        for lsq6_target in lsq6_targets:
            files_arg = csv_file if use_csv else files
            cmd = ["LSQ6.py", lsq6_target, lsq6_method] + files_arg + general_args + n3_arg + inorm_arg
            ret = script_runner.run(*cmd)
            assert ret.success

class TestLSQ12:
    # TODO find a way to parametrize this (i.e., vs deeply nested for-loops) that also works for MBM pipeline
    @pytest.mark.parametrize('use_csv', (True, False))
    def test(self, script_runner, files, csv_file, use_csv, lsq12_protocol):
        files_arg = csv_file if use_csv else files
        cmd = ["LSQ12.py", f"--lsq12-protocol={lsq12_protocol}"] + files_arg + general_args
        ret = script_runner.run(*cmd)
        assert ret.success

class TestNLIN:
    @pytest.mark.parametrize('registration_method', registration_methods)
    @pytest.mark.parametrize('registration_strategy', registration_strategies)
    @pytest.mark.parametrize('use_csv', (True, False))
    def test(self, script_runner, files, csv_file, use_csv, initial_model, registration_method, registration_strategy):
        files_arg = csv_file if use_csv else files
        cmd = ["NLIN.py", f"--target={initial_model}"] + files_arg + general_args + registration_method + registration_strategy
        ret = script_runner.run(*cmd)
        assert ret.success

class TestMAGeT:
    @pytest.mark.parametrize('use_csv', (True, False))
    #@pytest.mark.parametrize('registration_method', registration_methods)
    @pytest.mark.parametrize('maget_mask', (False, True))
    def test(self, script_runner, files, csv_file, use_csv, lsq12_protocol, nonlinear_protocol,
             atlas_dir, maget_mask):
        files_arg = csv_file if use_csv else files
        if maget_mask:
            mask_args = ['--mask', f'--masking-nlin-protocol={nonlinear_protocol}', '--masking-method=minctracc']
        else:
            mask_args = ["--no-mask"]
        cmd = (["MAGeT.py", f"--lsq12-protocol={lsq12_protocol}", f"--atlas-library={atlas_dir}",
                '--registration-method=minctracc', f'--nlin-protocol={nonlinear_protocol}']
               + files_arg + general_args + mask_args)
        ret = script_runner.run(*cmd)
        assert ret.success

class TestMBM:
    @pytest.mark.parametrize('lsq6_method', lsq6_methods)
    @pytest.mark.parametrize('registration_method', registration_methods)
    @pytest.mark.parametrize('registration_strategy', registration_strategies)
    @pytest.mark.parametrize('use_csv', (True, False))
    @pytest.mark.parametrize('run_maget', (True, False))
    def test(self, script_runner, files, csv_file, use_csv, run_maget, atlas_dir, nonlinear_protocol,
             initial_model, lsq6_method, lsq6_targets, lsq12_protocol,
             registration_method, registration_strategy):
        files_arg = csv_file if use_csv else files
        for lsq6_target in lsq6_targets:
            maget_args = [f"--maget-atlas-library={atlas_dir}"] if run_maget else ['--no-run-maget', '--maget-no-mask']
            cmd = (["MBM.py", lsq6_target] +
                    [f"--maget-nlin-protocol={nonlinear_protocol}",
                     f"--lsq12-protocol={lsq12_protocol}",
                     "--maget-registration-method=minctracc"] +
                    maget_args + files_arg + general_args + registration_method + registration_strategy)
            ret = script_runner.run(*cmd)
            assert ret.success

class TestTwoLevel:
    @pytest.mark.parametrize('use_csv', (True, False))
    def test(self, script_runner, files, csv_file, use_csv, initial_model, lsq12_protocol):
        files_arg = csv_file if use_csv else files
        pytest.xfail("not implemented")
        cmd = (["twolevel_model_building.py", f'--init-model={initial_model}', f"--lsq12-protocol={lsq12_protocol}"]
               + files_arg + general_args)
        ret = script_runner.run(*cmd)
        assert ret.success

class TestTamarack:
    def test(self, script_runner):
        pytest.xfail("not implemented")

class TestRegistrationChain:
    @pytest.mark.parametrize('common_time_point', chain_time_points)
    @pytest.mark.parametrize('use_pride_of_models', (True, False))
    def test(self, script_runner, lsq12_protocol, chain_csv_file, initial_model, common_time_point, pride_of_models, use_pride_of_models):
        initial_model_args = [f"--pride-of-models={pride_of_models}"] if use_pride_of_models else [f"--init-model={initial_model}"]
        cmd = ["registration_chain.py",
               f"--chain-csv-file={chain_csv_file}",
               f"--chain-common-time-point={common_time_point}",
               f"--lsq12-protocol={lsq12_protocol}"] + general_args + initial_model_args
        ret = script_runner.run(*cmd)
        assert ret.success

class TestAsymmetry:
    @pytest.mark.parametrize('registration_method', registration_methods)
    @pytest.mark.parametrize('use_csv', (True, False))
    def test(self, script_runner, files, csv_file, use_csv, registration_method, initial_model, lsq12_protocol):
        files_arg = csv_file if use_csv else files
        cmd = (["asymmetry.py", f"--lsq12-protocol={lsq12_protocol}",
                "--no-run-maget", "--maget-no-mask", f"--init-model={initial_model}"]
                + files_arg + general_args + registration_method)
        ret = script_runner.run(*cmd)
        assert ret.success
