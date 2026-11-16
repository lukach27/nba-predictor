NBA MATCH PREDICTOR - PROJECT LOG / README
============================================

CILJ PROJEKTA
-------------
Predviđanje pobjednika NBA utakmica (domaćin vs gost) koristeći
javno dostupne statistike (nba_api) i klasične ML modele
(Logistic Regression, XGBoost). Fokus je na razumljivom,
korak-po-korak pipelineu i mjerljivom poboljšanju kroz iteracije,
ne na "crnoj kutiji".


ARHITEKTURA / TOK PODATAKA
---------------------------
1. data_collection.py
   - Dohvaća sirove podatke o utakmicama s NBA API-ja (nba_api,
     endpoint leaguegamefinder) za sezone 2019-20 do 2024-25.
   - Sprema kao data/raw/games_raw.csv
   - Napomena: svaki GAME_ID ima 2 reda (jedan po timu). Filtrirano
     je da se izbace preseason/all-star utakmice (GAME_ID prefiks
     mora biti '002' = regular season).

2. prepare_features.py (feature engineering)
   - Učitava sirovi CSV, filtrira regular season.
   - Sortira po TEAM_ID + GAME_DATE (nužno za ispravan rolling).
   - Računa rolling prosjek zadnjih 5 utakmica po timu za:
     PTS, FG_PCT, REB, TOV, AST (koristi .shift(1) da izbjegne
     data leakage - ne smije uključivati statistiku same te utakmice).
   - Računa days_rest (dani odmora od prošle utakmice), capano na
     max 14 dana da izbjegne ekstremne vrijednosti na granici sezona.
   - Računa venue-specific rolling (forma tima ODVOJENO kad igra
     doma vs kad igra vani) - grupirano po (TEAM_ID, is_home).
   - Računa H2H (head-to-head) statistiku: prosjek razlike u
     poenima i home win rate iz zadnja 3 međusobna susreta tog
     para timova (matchup_key = sortirani spoj oba TEAM_ID-a, da
     je isti bez obzira tko je doma).
   - Računa ELO RATING sustav (K=20): kumulativni rejting snage
     tima koji se ažurira nakon svake utakmice na temelju ishoda
     i očekivanja (jača pobjeda protiv boljeg tima = veći skok
     rejtinga). Elo se računa kronološki, red po red, jer svaki
     rejting ovisi o prethodnom stanju.
   - Računa elo_trend (promjena Elo-a u zadnjih 10 utakmica tima,
     doma+vani zajedno preko "dugog formata") i is_back_to_back
     (flag za <=1 dan odmora) - izračunati i spremljeni u CSV, ali
     izbačeni iz X seta za model (vidi "Testirano i odbačeno").
   - Računa _diff kolone (home - away) za sve gornje statistike -
     olakšava modelu (pogotovo linearnom) da vidi relativnu
     razliku bez da je sam mora "izračunati".
   - Spaja home/away perspektivu svake utakmice u jedan red
     (merge po GAME_ID, suffixes=('_home', '_away')).
   - Izbacuje redove s NaN (prve utakmice svakog tima/para koje
     nemaju dovoljno povijesti za rolling prozore).
   - Sprema kao data/processed/games_merged.csv

3. train_model.py (modeliranje)
   - Cilj (y): home_win (1 ako je domaćin pobijedio, 0 inače).
   - Featurei (X): SAMO kolone koje su poznate PRIJE utakmice
     (rolling_5_*, days_rest, *_diff, elo_home/away/diff, h2h_*)
     - NIKAD sirovi rezultat te utakmice (PTS_home, WL, itd.) jer
     bi to bio data leakage.
   - Split: VREMENSKI (sortirano po datumu, prvih 70% train,
     sljedećih 10% validation, zadnjih 20% test) - NE nasumičan,
     jer model mora predviđati budućnost, ne nasumične utakmice
     iz cijele povijesti.
   - Modeli: XGBoost (XGBClassifier) i Logistic Regression, oba s
     random_state=42 radi reproducibilnosti rezultata.
     Logistic Regression treba StandardScaler (skaliranje featurea)
     jer su featurei na različitim skalama (FG_PCT ~0.45 vs
     PTS ~110) - bez skaliranja ne konvergira ispravno.
   - Hiperparameter tuning: GridSearchCV s TimeSeriesSplit(5) (ne
     standardni CV, jer bi nasumično miješao vremenski redoslijed).
     Scoring = neg_brier_score (optimizira kalibraciju, ne samo
     pogotke). Grid: n_estimators [200,300,500], max_depth [2,3,4],
     learning_rate [0.005,0.01,0.05]. Najbolji: n_estimators=300,
     max_depth=2, learning_rate=0.01.
   - Evaluacija: accuracy, Brier score (mjeri kalibraciju
     vjerojatnosti, ne samo pogotke), usporedba s naivnim
     baseline-om ("uvijek predvidi home win").


REZULTATI KROZ ITERACIJE
--------------------------
Verzija                          | XGB Acc | XGB Brier | LogReg Acc | LogReg Brier
----------------------------------|---------|-----------|------------|-------------
Baseline (samo rolling_5)         | 0.595   | 0.236     | 0.604      | 0.233
+ days_rest                       | 0.605   | 0.234     | -          | -
+ diff kolone                     | 0.600   | 0.235     | 0.617      | 0.231
+ venue-specific rolling          | 0.606   | 0.230     | 0.622      | 0.229
+ head-to-head (H2H)              | 0.627   | 0.230     | 0.622      | 0.227 (skalirano)
+ ELO RATING                      | 0.656   | 0.215     | 0.659      | 0.214
+ hiperparameter tuning (FINALNO) | 0.662   | 0.215     | 0.660      | 0.214

Naivni baseline (uvijek "home win"): ~0.541-0.546 accuracy.

Napomena: raniji rezultat od 0.668 (prije uvođenja random_state)
nije reproduciran nakon fiksiranja seeda - vjerojatno je bio
sretan slučajni rezultat, ne stvarno bolja konfiguracija. 0.662
je pouzdana, reproducibilna finalna brojka.


KLJUČNI NALAZI
----------------
- ELO RATING je ubjedljivo najjači pojedinačni feature (importance
  ~0.20-0.21, gotovo 4x veći od sljedećeg najboljeg) - kumulativni
  rejting kroz cijelu povijest nosi jači signal od bilo koje
  statistike zadnjih 5 utakmica.
- FG_PCT_diff (razlika u efikasnosti šuta) je bio dosljedno najjači
  "obični" feature prije uvođenja Elo-a - jači prediktor od sirovih
  poena ili rebound razlika.
- Više featurea NE znači automatski bolji model - XGBoost je
  ponekad blago padao s dodavanjem featurea (npr. diff kolone) dok
  je Logistic Regression od njih profitirao. Svaki dodatak je
  testiran i zadržan/odbačen na temelju mjerenih rezultata, ne
  pretpostavke.
- Data leakage je stalna opasnost u pipelineu ovog tipa - sprječen
  s .shift(1) prije svakog rolling izračuna i strogim pravilom da
  X nikad ne smije sadržavati "rezultat" utakmice koja se predviđa.
  Fiksni random_state je bitan za pouzdanu usporedbu iteracija -
  bez njega, XGBoost varijanca (~0.005-0.01 accuracy) lako se
  zamijeni za stvarno poboljšanje/pogoršanje.
- S max_depth=2 (odabran GridSearch-om), dominacija elo_diff
  featurea gura importance nekih drugih kolona (REB_home, TOV_diff,
  AST_diff, H2H statistike) na točno 0.0 - plitka stabla nemaju
  "mjesta" za manje-informativne featurese kad postoji jedan jako
  dominantan. Ne znači da su ti featurei beskorisni sami za sebe,
  samo da ih ovaj konkretan model ne koristi.


TESTIRANO I ODBAČENO
-----------------------
- elo_trend (promjena Elo-a u zadnjih 10 utakmica, doma+vani
  zajedno): nakon ispravke bug-a u X filteru i dodavanja fiksnog
  seeda, nije poboljšao metrike (0.662 s vs bez, praktički
  identično) - vjerojatno duplicira signal koji trenutni elo_diff
  već nosi. Izračun zadržan u CSV-u za potencijalni prikaz "tim u
  usponu" na dashboardu, ali izbačen iz X seta za model.
- is_back_to_back (binarni flag za <=1 dan odmora): nije pokazao
  dodatnu vrijednost preko postojećeg kontinuiranog days_rest -
  vjerojatno prekratak/rijedak signal da ga plitka XGBoost stabla
  (max_depth=2) uhvate. Izbačen iz X seta.


POZNATA OGRANIČENJA / IDEJE ZA DALJE
---------------------------------------
- Nema podataka o ozljedama/izostancima ključnih igrača (nba_api
  nema čist endpoint za to) - potencijalno najveći preostali izvor
  greške u modelu.
- H2H statistika ima relativno malen uzorak po paru timova (2-4
  susreta godišnje) - signal je slabiji nego kod ostalih featurea.
- Realan strop za NBA predikciju je otprilike 65-70% accuracy zbog
  inherentne slučajnosti sporta - čak i profesionalni modeli s puno
  bogatijim podacima rijetko prelaze tu granicu. 0.662 je solidan
  rezultat unutar tog realnog okvira.
- Sljedeći koraci: LLM tactical agent sloj (generira tekstualni
  "pre-match breakdown" iz model outputa), interaktivni dashboard
  (Streamlit) za pregled utakmica i vizualizaciju forme timova.

Ključni nalazi": "Ispravka H2H logike (perspektivno okretanje predznaka kad se domaćinstvo mijenja između susreta) podigla je accuracy s 0.662 na 0.667 i pretvorila H2H featurese iz potpuno neaktivnih (importance 0.0) u top-10 najvažnije — pokazatelj da je bug u feature engineeringu, ne nedostatak signala, bio uzrok ranije niske korisnosti.

STATUS: FEATURE ENGINEERING I MODEL - ZAVRŠENO (za sad)
Sljedeća faza: LLM Tactical Agent