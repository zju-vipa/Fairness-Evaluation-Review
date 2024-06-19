import sys
import os
import datetime

class Recorder:

    def __init__(self) -> None:
        self.log_line = 0

    def get_args(self, sys_argv):
        arg_ = 'python '
        note_ = ''
        for i, arg in enumerate(sys_argv):
            if 'note' in arg:
                note_ = sys_argv[i + 1]
                i = i + 1
            else:
                arg_ = arg_ + arg + ' '
        if sys_argv[0][0] == '/':
            note_ = 'debug_' + sys_argv[0].rsplit("/", maxsplit=1)[-1]
        for j in range(24 - len(note_)):
            note_ = note_ + ' ' 
        return arg_, note_

    def init_file(self, log_file):
        if not os.path.isfile(log_file):
            with open(log_file, 'w') as file:
                file.write('-------------------------------------------------------------\n')
                file.write('| Start Time           | Arguments                          |\n')
                # file.write('-------------------------------------------------------------\n')
                file.write('\n')
            os.chmod(log_file, 0o777)

    def update_log(self, log_file, start_time, arg_):
        existing_content = []
        with open(log_file, 'r') as file:
            existing_content = file.readlines()
        self.log_line = len(existing_content) -1
        existing_content.insert(self.log_line, f'|*{start_time} | {arg_} |\n')
        if not existing_content[self.log_line-1][2:12] == str(start_time)[0:10]:
            existing_content.insert(self.log_line, '-------------------------------------------------------------\n')
            self.log_line += 1
        with open(log_file, 'w') as file:
            file.writelines(existing_content)
        print(f'Experiment log saved to {log_file} successfully.')

    def check_finished(self, log_file):
        existing_content = []
        with open(log_file, 'r') as file:
            existing_content = file.readlines()
        self.log_line = len(existing_content) -1
        existing_content[self.log_line - 1] = existing_content[self.log_line - 1].replace('*', ' ')
        with open(log_file, 'w') as file:
            file.writelines(existing_content)

    def record_log(self, log_file = 'exp_log.txt', sys_argv = None):
        start_time = datetime.datetime.now().replace(microsecond=0)
        self.init_file(log_file)
        arg_, note_ = self.get_args(sys_argv)
        self.update_log(log_file, start_time, arg_)
