from ObservationTable import ObservationTable
import DFA
from time import perf_counter

def run_lstar(teacher,time_limit):
    table = ObservationTable(teacher.alphabet,teacher)
    start = perf_counter()
    teacher.counterexample_generator.set_time_limit(time_limit,start)
    table.set_time_limit(time_limit,start)

    while True:
        while True:
            while table.find_and_handle_inconsistency():
                pass
            if table.find_and_close_row():
                continue
            else:
                break
        dfa = DFA.DFA(obs_table=table)
        print("obs table refinement took " + str(int(1000*(perf_counter()-start))/1000.0) )
        counterexample = teacher.equivalence_query(dfa)
        if None is counterexample:
            break
        start = perf_counter()
        table.add_counterexample(counterexample,teacher.classify_word(counterexample))
    return dfa