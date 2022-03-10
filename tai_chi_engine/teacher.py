from tai_chi_tuna.front.teacher import Teacher, teach
import tai_chi_engine
from pathlib import Path

class EngineTeacher(Teacher):
    """
    The teacher decorator using tai_chi_engine's static folder as starting point

    @EngineTeacher("explain_to_some_class.html")
    class SomeClass:
        def __init__(self, ...)
    """
    
    static_folder = Path(tai_chi_engine.__file__).parent/"static"