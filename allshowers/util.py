import datetime
import os
import shutil

__all__ = ["setup_result_path"]


def setup_result_path(run_name: str, conf_file: str, fast_dev_run: bool = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)

    now = datetime.datetime.now()
    while True:
        full_run_name = now.strftime("%Y%m%d_%H%M%S") + "_" + run_name
        result_path = os.path.join(repo_dir, "results", "allShowers", full_run_name)
        if not os.path.exists(result_path):
            if not fast_dev_run:
                os.makedirs(result_path)
            else:
                result_path = os.path.join(repo_dir, "results", "allShowers", "test")
                if os.path.exists(result_path):
                    shutil.rmtree(result_path)
                os.makedirs(result_path)
            break
        else:
            now += datetime.timedelta(seconds=1)

    with open(conf_file) as f:
        content_list = f.readlines()

    content_list = [line for line in content_list if not line.startswith("result_path")]
    content_list.insert(1, f"result_path: {result_path}\n")
    content = "".join(content_list)

    with open(os.path.join(result_path, "conf.yaml"), "w") as f:
        f.write(content)

    return result_path
