import pandas as pd
from pyswip import Prolog # type: ignore

# Load the patientes_dataset.csv dataset
dataset = pd.read_csv("./Dataset/patients_dataset.csv")

# Prolog rules to write to file
prolog_rules = """

% Calcular factores de riesgo normalizados
calculate_factors(LymphocytesT, LymphocytesB, RedCells, Background, Factors) :-
    NormalizedLymphocytesT is LymphocytesT / 1300, 
    NormalizedLymphocytesB is LymphocytesB / 2600,
    NormalizedRedCells is RedCells / 6.7, 
    FactorT is NormalizedLymphocytesT * 0.4, 
    FactorB is NormalizedLymphocytesB * 0.3,
    FactorR is NormalizedRedCells * 0.2,
    FactorBG is Background * 0.1,
    Factors = [FactorT, FactorB, FactorR, FactorBG].

% Calcular riesgo total
calculate_risk(Factors, Risk) :-
    [FactorT, FactorB, FactorR, FactorBG] = Factors,
    Risk is FactorT + FactorB + FactorR + FactorBG.

% Determinar enfermedades posibles (evaluación independiente de las condiciones)
possible_diseases(RedCells, LymphocytesT, LymphocytesB, Risk, PossibleDiseases) :-
    findall(Disease,
           (   (RedCells < 4.5, Disease = anemia);
               (LymphocytesT < 500, LymphocytesB < 600, Disease = inmunodeficiencia);
               (Risk >= 0.8, Disease = linfoma);
               (Risk >= 0.5, Risk < 0.8, Disease = leucemia);
               (Risk > 0.3, Risk < 0.5, Disease = 'posible cancer')
           ),
           RawDiseases),
    sort(RawDiseases, PossibleDiseases). % Eliminar duplicados y ordenar.


% Determinar diagnóstico final (considerando prioridades y excluyendo redundancias)
disease_type(LymphocytesT, LymphocytesB, RedCells, Background, Diagnoses) :-
    calculate_factors(LymphocytesT, LymphocytesB, RedCells, Background, Factors),
    calculate_risk(Factors, Risk),
    possible_diseases(RedCells, LymphocytesT, LymphocytesB, Risk, PossibleDiseases),
    sort(PossibleDiseases, Diagnoses). % Eliminar duplicados y ordenar


"""

# Open the KB_cancer.pl file in append mode and write the Prolog rules
with open("KB_cancer.pl", "a") as prolog_file:
    prolog_file.write(prolog_rules)


# Create a Prolog object
prolog = Prolog()

# Load the KB_cancer.pl file
prolog.consult("KB_cancer.pl")