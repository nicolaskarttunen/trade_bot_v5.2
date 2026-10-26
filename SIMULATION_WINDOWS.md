# Paikallinen offline-simulaatio Windows PowerShellissä

Tämä käynnistää automaattitestit ja V6.5-moottorin **keinotekoisella datalla**.
Alpaca-paperitiliä tai API-avaimia ei tarvita. Oracle-bottia ei käytetä eikä
käynnissä olevan botin hakemistoon tehdä muutoksia. Simulaatio ei ole reaaliaikainen
paper-ajo eikä sadan osakkeen kannattavuustesti.

## Asennus ja ajo

Tarvitaan Git ja Python **3.12**. Tarkista Python ensin:

```powershell
py -3.12 --version
```

Jos komentoa tai versiota ei löydy, asennus pitää selvittää ennen jatkamista.
`py -0p` näyttää Windowsin Python-asennukset. Testattu ympäristö on Python 3.12;
älä vaihda sitä huomaamatta nykyisen botin Python-versioon.

Suorita tämä yhtenä lohkona PowerShellissä. Se luo aikaleimalla nimetyn uuden
hakemiston Downloads-kansioon ja keskeyttää ensimmäiseen komentovirheeseen.
Virtuaaliympäristöä ei aktivoida, joten PowerShellin suorituspolitiikkaa ei tarvitse muuttaa.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $simDir = Join-Path $env:USERPROFILE ("Downloads\tradebot-simulaatio-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    git clone --branch audit-hardening-v1 --single-branch https://github.com/nicolaskarttunen/trade_bot_v5.2.git $simDir
    if ($LASTEXITCODE -ne 0) { throw 'Git clone failed' }
    Set-Location $simDir
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
    .\.venv\Scripts\python.exe -m pip install -r requirements-simulation.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    $env:OPENBLAS_NUM_THREADS = '1'
    .\.venv\Scripts\python.exe -X utf8 tools\run_offline_simulation.py --output-dir research_runs\ensimmainen
    if ($LASTEXITCODE -ne 0) { throw 'Simulation failed; inspect the log above' }
}
```

Git-kloonaus ja riippuvuuksien asennus käyttävät internetiä. Varsinaisen
testisarjan ja demon verkkoyhteydet estetään. `-X utf8` on tarkoituksella mukana
Windowsin merkistöeroja varten; myös testialiprosessi käynnistyy UTF-8-tilassa.

Uusi ajo samassa hakemistossa:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\run_offline_simulation.py --output-dir research_runs\toinen
```

Vanhoja tuloshakemistoja ei ylikirjoiteta. Älä käynnistä tässä kokeessa
`paper_trader_integrated.py`:tä. Simulaation käynnistin ei tarvitse `.env`-tiedostoa
ja kouluttaa demon mallin muistissa muuttamatta `models/ensemble.pkl`:ää.

## Mitä tulokset tarkoittavat

Ensimmäinen vaihe ajaa nyt **108 automaattista testiä**, mukaan lukien toimeksiantojen
kadonneet kuittaukset, restart, osatäytöt ja riskirajat. Nämä käyttävät
valevälittäjää. Varsinainen OHLC-simulaattori ei mallinna samoja välittäjätapahtumia.

Toinen vaihe ajaa oikean V6.5-ajurin kahdella keinotekoisella osakesarjalla
(NVDA/AMD-nimet) ja keinotekoisella SPY-sarjalla. Malli koulutetaan kahdeksan
keinotekoisen päivän datalla ja ennustaa yhdeksän myöhempää keinotekoista päivää.
Kaksi vertailuversiota ovat nykyiset kiinteät asetukset ilman SPY-porttia ja
SPY +0,10 % -portilla. Riskiasetuksia, ML-rajoja tai mallin koulutusparametreja
ei muuteta; vain demon osakejoukko korvataan prosessin muistissa ja palautetaan.

Demon SPY-sarjan jokaisen kynttilän avaus ja sulku ovat samat. Siksi SPY-portti
estää kaikki kaupat. Tämä on fixture-datan ominaisuus, **ei havainto suodattimen
hyödystä tai haitasta**. Myöskään ilman porttia syntyvät tuotot eivät ole
markkinanäyttöä. Tämä pieni demo ei kata kaikkia riskitilanteita.

Tulokset löytyvät `research_runs\ensimmainen`-hakemistosta:

- `tests/test_results.json` ja `.txt`: testitulokset sekä lähdekooditiivisteet.
- `demo/run_report.json`: onnistuminen, datan synteettisyys ja kauppamäärät.
- `demo/driver.log`: V6.5-ajurin loki.
- `demo/replay/v6_5_intraday_trades.csv`: simuloidut kaupat.
- `demo/replay/v6_5_intraday_daily.csv`: päiväkohtaiset laskelmat.
- `demo/replay/v6_5_intraday_equity.csv`: simuloitu pääoman kehitys.
- `demo/replay/manifest.json`: käytetyt asetukset ja tiedostotiivisteet.
- `demo/READ_ME_FIRST.txt`: tulkintarajat tulosten mukana.

Demo tuottaa CSV-raportteihin myös laskennallisia tuottoja. Niitä ei pidä esittää
botin historiallisina tuottoina. Demotulokset eivät ole strategian kehityskokeita.

## Varmennus ja seuraava oikeaan dataan perustuva testi

Käynnistin tarkistettiin Linuxissa Python 3.12:lla; koko komentopolku ajoi 100/100
testiä ja valmistui yhdeksän päivän / kahden vertailuhaaran demoon. Sama testisarja
ajettiin pandas 2.2.3:lla ja 3.0.0:lla. Windows-tarkistus on lisätty GitHub Actionsiin (Windows/Linux × pandas 2.2.3/3.0.0),
mukaan lukien koko käynnistyskomento. Tarkista ajon toteutunut tila Actionsista;
käyttäjän omalla Windows-koneella ajo on vielä varmistamatta. Uudet kolme testiä tarkistavat demon merkinnät,
manifestin tiivisteet, asetusten palautumisen, epäonnistumisen kirjaamisen ja
vanhojen tulosten säilymisen.

GitHubissa nyt tarkistettu historiallinen aineisto kattaa seitsemän osaketta
sekä SPY/QQQ:n elokuun 2026 loppuun asti. Nykyinen varsinainen 100 osakkeen
V6.5-vertailu vaatii kaikkien määritettyjen osakkeiden historian, syys–lokakuun
laajennustiedostot ja pörssin istuntokalenterin. Puuttuvia osakkeita ei korvata
hiljaisesti synteettisillä sarjoilla eikä jätetä pois tuloksen parantamiseksi.

Oikeaan dataan perustuvan kokeen seuraava vaihe on paikallisten historiatiedostojen
inventointi. Katsotut jaksot merkitään kehitysaineistoksi; parametreja ei säädetä
jälkikäteen parantamaan tulosta. Tuore etukäteen määritetty arviointijakso ja
uutisten vastaanottoajat tarvitaan erilliseen luotettavaan hyötyarvioon.

Tämä offline-demo ei poista [toimeksiantoauditoinnin](ORDER_LIFECYCLE_AUDIT_2026-10-26.md)
käyttöönottoesteitä: erityisesti päivän lopun oikean välittäjäsulun epäselvät
kuittaukset ovat edelleen korjaamatta.

Paikalliset ajolokit: [pandas 2.2.3](audit_evidence/2026-10-26-simulation/pandas-2.2.3/test_results.json), [pandas 3.0.0](audit_evidence/2026-10-26-simulation/pandas-3.0.0/test_results.json) ja [demon tilaraportti](audit_evidence/2026-10-26-simulation/demo_report.json).

## Windows-tarkistuksessa löytyneet korjaukset

Ensimmäinen Windows-CI-ajo ([40346579696](https://github.com/nicolaskarttunen/trade_bot_v5.2/actions/runs/40346579696))
paljasti molemmilla pandas-versioilla kaksi virhettä, vaikka Linux-ajot onnistuivat:

- Prosessilukon luonti yritti lukea toisen kahvan lukitsemaa tavua ennen
  varsinaista lukitusyritystä. Windows esti tämän jo lukuoperaatiossa.
  Tiedoston koko tarkistetaan nyt `fstat`-kutsulla lukittua aluetta lukematta,
  ja alustus kuuluu samaan virheenkäsittelyyn kuin lukitus.
- SQLite-yhteyden `PRAGMA`-alustus oli ennen `try/finally`-lohkoa.
  Vioittuneen tietokannan alustusvirhe jätti kahvan auki, jolloin Windows ei
  pystynyt poistamaan testin väliaikaistiedostoa. Myös alustus kuuluu nyt
  lohkoon, joka sulkee yhteyden varmasti.

Lisätty regressiotesti varmistaa tietokantakahvan sulkemisen alustusvirheessä.
Testisarjassa on nyt 100 testiä. Nämä ovat käyttöjärjestelmä-/virheenkäsittelykorjauksia,
eivät strategiamuutoksia. Vanhaa lokkia ei poisteta tai nollata korjauksena.

## Oikean historian aukkojen selvitys

Käyttäjän 102 tiedoston SIP-aineistosta löydettyjen viiden aukon erillinen,
vain hintadataa lukeva tarkistus: [GAP_PROBE_2026-10-26.md](GAP_PROBE_2026-10-26.md).
Kahdeksan lisätestiä nostavat nykyisen testimäärän 108:aan. Varsinainen
synteettinen toimintademo ja strategian asetukset eivät muutu.
