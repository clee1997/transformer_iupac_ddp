import jpype
import os

def get_opsin_res(str_in):

    # opsin_jar_path = os.path.join(project_path, 'opsin-core-2.7.0-jar-with-dependencies.jar')

    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=opsin_jar_path)

    jpkg = jpype.JPackage('uk.ac.cam.ch.wwmm.opsin')
    NameToStructure = jpkg.NameToStructure # does work 
    nts = NameToStructure.getInstance()

    parsing = nts.parseChemicalName(str_in)

    opsin_res = parsing.getStatus()
    opsin_res = bool(opsin_res == "SUCCESS")

    return opsin_res