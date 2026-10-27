# Elokuun 2026 kehityssimulaatio

Vain haara `audit-hardening-v1`. Ei tuotantoon yhdistämistä, Oracle-palvelimen
muutosta tai toimeksiantoja. Tämä ajuri on erillinen tutkimustyökalu;
strategian parametreja, alkuperäistä syys–lokakuun ajuria ja `config.json`:ia
ei muutettu.

## Miksi elokuu ja mitä aukoista tiedetään?

Käyttäjän Windows-koneella on 102 hintatiedostoa, joiden yhteinen loppu on
31.8.2026. Syys–lokakuun ajurin vaatimaa jatkoaineistoa ei ole käytettävissä.
Elokuu valittiin saatavuuden perusteella ennen tämän korjatun ajon tuottojen
tarkastelua. Se ei silti ole uusi riippumaton holdout: historiaa ja strategiaa
on tutkittu aiemmin. Ajanjaksoa ei siirretä tuloksen perusteella.

Käyttäjän raportoima SIP/raw-uudelleenhaku: EQIX:n neljä ja BLK:n yksi elokuun
kynttilä puuttuivat myös vastauksesta. Kaikissa viidessä ikkunassa neljä
ympäröivää OHLCV-riviä täsmäsivät (20/20), tarkistus valmistui ja lähdetiedostojen
SHA-256:t säilyivät. Tämä tukee paikallisen aineiston vastaavuutta kyseisissä
ikkunoissa, mutta ei selvitä aukkojen syytä tai todista koko aineiston
syötettä, oikaisuja tai täydellisyyttä. Assistentti ei ole lukenut käyttäjän
koko 102 tiedoston aineistoa eikä ajanut sillä tuottosimulaatiota.

## Kiinteä koe

- Koulutus: käytettävissä oleva historia 1.1.2023–31.7.2026. Labelin pitää olla
  kokonaan saatavilla viimeistään 1.8.2026 klo 00.00 New Yorkissa. Koulutus käyttää
  yhteistä `training_rows`/`fit_bundle`-polkua ja samaa 250 havainnon piirrelogiikkaa.
- Testi: 1.–31.8.2026, kalenterissa 21 sessiota 3.–31.8.2026, kaikki klo
  09.30–16.00 ET. Kalenteri ei muodostu SPY:stä. Testin 1638 odotettua kynttilää
  osaketta kohti johdetaan erillisestä kalenterista.
- Kaikki 100 määriteltyä osaketta alkuperäisessä järjestyksessä sekä SPY.
  QQQ-tiedostoa ei lueta: nykyinen yhteinen malli käyttää vain ensimmäistä
  benchmark-symbolia eli SPY:tä. Tiedostovaatimus on siksi 101, ei 102.
- Kaksi alkuperäisen V6.5-ajurin kiinteää haaraa: ei porttia / SPY:n ensimmäisen
  valmistuneen 5 min kynttilän tuotto vähintään +0,10 %. Yksi yhteinen koulutettu
  malli, samat hinnat ja kulut molemmille. Ei parametrien hakua.
- `feed=sip` asetetaan vain ajurin omaan muistissa olevaan konfiguraatioon
  tutkimusaineiston oletuksena. Täyttä alkuperätodistusta ei väitetä. SIP-mallia
  ei siirretä IEX-paperibotin malliksi. Uutiset neutraalit, OPG/premarket-toimeksiannot
  pois; saatavilla olevat pre-/postmarket-havainnot säilyvät piirteiden historiassa
  kuten yhteisessä koulutuspolussa. Myöhempi syyskuun data rajataan pois.
- Mallin koulutusta varten säilytetään vain aikaleima, piirteet ja target.
  Aikaleimajärjestys on vakaa. Koulutus ei ota satunnaista pienempää otosta tai
  muuta ensemble-mallien määriä/iteraatioita ajon nopeuttamiseksi.

## Aukkojen käsittely

Kaikkien osakkeiden jokaisen elokuun session pitää löytyä. SPY:ltä vaaditaan
kaikki 1638 normaalin session havaintoa. Puuttuvat osakekynttilät listataan,
mutta niitä ei täytetä, osakkeita/päiviä ei poisteta eikä tulevan aukon vuoksi
kielletä aikaisempaa sisäänmenoa.

V6.5:n nykyinen tiukka suoritusmalli pysyy voimassa: avoimen position
hintakynttilän puuttuminen keskeyttää ajon. Myös EOD-hinnan pitää löytyä.
Jos kumpi tahansa haara epäonnistuu, vertailulle ei kirjoiteta hyväksyttyjä
tuottotaulukoita tai SPY-vertailua. Puuttuvan nykyisen sisäänmenokynttilän
kohdalla ei simuloida täyttöä. Uutisia tai tick-tason stop-toteutuksia ei
päätellä kynttilöiden väliltä.

## Windows: tarkistus ennen raskasta koulutusta

Suorita erillisessä simulaatiokansiossa, jossa `.venv` ja riippuvuudet ovat jo
asennettu. Työkalu ei lue `.env`-tiedostoja, vanhoja model-picklejä tai API-avaimia.
Verkkoyhteydet estetään ajon ajaksi. Olemassa olevaa tuloskansiota ei korvata.

```powershell
git pull --ff-only origin audit-hardening-v1
if ($LASTEXITCODE -ne 0) { throw 'Git update failed' }

.\.venv\Scripts\python.exe -X utf8 tools\run_august_simulation.py `
    --data-dir "$env:USERPROFILE\Downloads\trade_bot_v5.2\data_sip" `
    --output-dir research_runs\elokuu-tarkistus-1
```

Tämä oletusajo tarkistaa tiedostot, kalenterikattavuuden, puuttuvat kynttilät
sekä aineiston koon. Se EI kouluta tai laske tuottoa. Raportti ilmoittaa
raakakoulutusrivien ylärajan ja yhden float64-piirrematriisin koon; tämä EI ole
RAM-huipun arvio. Koulutus käyttää useita matriiseja ja kopioita, joten todellinen
muistitarve on tätä huomattavasti suurempi. Käyttäjän laitteiston koko aineiston
muistinkulutusta tai kestoa ei ole mitattu.

Varsinainen kiinteä ajo tehdään lisäämällä `--run` ja käyttämällä uutta kansiota:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\run_august_simulation.py `
    --data-dir "$env:USERPROFILE\Downloads\trade_bot_v5.2\data_sip" `
    --output-dir research_runs\elokuu-simulaatio-1 `
    --run
```

Tulokset: `run_report.json`, `manifest.json` sekä vasta molempien haarojen
onnistuttua `summary.csv`, `daily.csv`, `trades.csv`, `equity.csv` ja
`spy_comparison.json`. Manifesteihin kirjataan lähteiden, kalenterin ja koodin
tiivisteet sekä ajon tila. Lähteiden muuttuminen kesken ajon hylkää tuloksen.
Keskeytettyyn ajoon jää `failed_not_validated`; Ctrl+C käsitellään samoin.

## Oikeasti testattu

Kahdeksan uutta testiä keinotekoisella kahden osakkeen + SPY:n aineistolla:

1. Pelkkä tarkistus säilyttää aukot, alkuperäiset ja vertailuajurin configin;
   se ei käynnistä koulutusta.
2. Puuttuva kokonainen osakesessio tai SPY-kynttilä estää koulutuksen.
3. Myöhempien testihintojen muutos ei muuta koulutusrivejä tai aikaisempia
   ennustepiirteitä; syyskuun rivit rajataan pois.
4. Oikea ensemble ja replay tuottavat kaksi 21 session sarjaa myös
   nollakauppapäiville; tulostiedostojen tiivisteet täsmäävät.
5. Pakotettu avoin positio + puuttuva kynttilä keskeyttää koko vertailun;
   tuottotaulukkoa ei julkaista.
6. Koulutuksen aikana muuttunut lähde estää tuloksen hyväksymisen.
7. Koulutuksesta yritetty verkkoyhteys estetään.
8. Ylikirjoitus, tuloskansio lähdedatan sisällä, puuttuva tiedosto ja vajaa
   kalenteri hylätään.

Kaikki testit ovat toiminta-/virhetestejä, eivät todiste kannattavuudesta.
Koko sarjassa on 116 testiä. Paikalliset tulokset tallennetaan hakemistoon
`audit_evidence/2026-10-26-august-simulation/`. Windows/Linux- ja pandas 2.2.3/3.0.0
-tulokset löytyvät kyseisen commitin GitHub Actions -ajosta.

Täysi 100 osakkeen historiallinen ajo, uutisten lisäarvo, corporate actions,
historiallinen point-in-time-osakejoukko ja välittäjätoteutukset ovat edelleen
vahvistamatta. Paperibotin EOD-/kuittausesteitä ei korjata tai oteta käyttöön
tämän tutkimustyökalun yhteydessä.

## Kalenterilähteet (tarkistettu 26.10.2026)

NYSE/ICE:n kalenteri ei listaa elokuulle 2026 pörssipyhiä tai lyhennettyjä
sessioita. Kuukauden normaalit arkipäivät on tallennettu tiedostoon
`research_calendars/nyse_2026_08.csv`. Tämä ei ole yleiskalenterin korvike.

- https://www.nyse.com/markets/hours-calendars
- https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf
- https://ir.theice.com/press/news-details/2024/NYSE-Group-Announces-2025-2026-and-2027-Holiday-and-Early-Closings-Calendar/default.aspx
