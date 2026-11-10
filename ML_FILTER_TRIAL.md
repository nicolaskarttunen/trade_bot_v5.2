# Vaikuttaako osakkeen ML-kelpoisuus tuottoon?

Tutkimuskoe 10.11.2026, erillään neljän haaran rekisteröidystä seurannasta. Lähtökohta: [kustannukset ja mallin tavoite](STABILITY_REVIEW.md). Aiemmin tarkastellut jaksot: elokuun 21 sessiota ja 20.10.–6.11.2026:n 14 sessiota. Molemmat raportoidaan erikseen ja kaikki päivät pidetään mukana. Tämä ei ole riippumaton kannattavuustesti.

- **original_ml:** alkuperäinen kiinteä osakejärjestys ja pisteytys ilman SPY:n erillistä +0,10 %:n avaussuodatinta. Neutraaleilla uutisilla numeerinen ostoehto vastaa ehtoa ml_prob ≥ 0,55 ja SPY:n kolmen viimeisen havaintovälin tuotto > 0.
- **market_only:** sama järjestys ja sama lyhyt markkinaehto; osakepisteet eivät rajoita ostoja. Lyhyt markkinaehto lasketaan samalla SPY:n viiden minuutin päätöshintasarjalla kuin alkuperäinen piirre. Puuttuvia havaintoja ei täytetä. Ennen vertailua tarkistetaan jokaisen alkuperäisen signaalin markkinatuottoa vastaava pisteluku. Molemmat haarat käyttävät vain samoja jo valmiita, piirteiltään kelvollisia signaalirivejä; kyseessä ei ole yleiskäyttöinen uusi malliton botti. Matala todellinen ennuste säilyy myös kontrollin kauppalokissa. Alkuperäiset riskirajat, kellot, uutiset ja kustannukset säilyvät; kassapolut ja täyttyvät paikat voivat erota.

Ensisijainen mitta on ML-version päivätuotto miinus markkinaversion päivätuotto kulujen jälkeen (bp/päivä; 5 päivän lohkot, 2 000 toistoa, 95 %:n kuvaileva väli). Positiivinen erotus suosii ML-kelpoisuutta. Lisäksi: nettotuotto, MTM-arvonlasku, kauppamäärä, päivävaihto (osto- ja myyntiarvo yhteensä / päivän alkupääoma), aikapainotettu altistusarvio (ostohintainen nimellisarvo × kirjattu pitoaika / koko session kesto / päivän alkupääoma), tulos +3 bp:n lisäkululla per puoli ja muiden päivien summa ilman parasta. Altistus on päiväkeskiarvo, joka sisältää nollapäivät; se ei kuvaa tick-tason markkina-arvoa. Kulustressi pitää kunkin haaran kaupat kiinteinä. Tätä ei tulkita riskivakioiduksi alfaksi eikä tehdä asetuspäätöstä paremman historiallisen tuoton perusteella.

## Paikallinen ajo

Aja samasta simulaatiokansiosta:

```powershell
& {
    if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Vaara kehityshaara' }
    git pull --ff-only origin audit-hardening-v1
    if ($LASTEXITCODE -ne 0) { throw 'Git-paivitys epaonnistui' }
    $out = 'research_runs/ml-vertailu-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    .\.venv\Scripts\python.exe -u -X utf8 tools\trial_ml_filter.py --output-dir $out
}
```

Oletukset: aiemmat elokuun järjestyskokeen reference- ja lukitun jakson kansiot sekä forward-20261110:n jäädytetty malli ovat paikoillaan. Elokuun ainoa erillinen hintasyöte on Downloads/trade_bot_v5.2/data_sip/SPY_5min.csv. Hakemiston voi osoittaa --history-valinnalla; SPY:n on täsmättävä alkuperäiseen tarkistussummaan. Loppujakson pidennetty SPY-historia sisältää myös piirteiden lämmittelyn. Ei uudelleenkoulutusta, API-tunnuksia tai verkkoa; malli ladataan alkuperäisten ennusteiden toistamiseen. Forward-kokeesta ei lueta päivä- tai tuottotuloksia. Alkuperäisen haaran kaupat, päivät, pääomakäyrät ja yhteenveto vaaditaan molemmissa jaksoissa vertailuaineiston mukaisiksi ennen hyväksyntää. Muuttunut syöte, avoimen position hintapolun aukko tai kontrollipoikkeama estää tuloksen hyväksymisen.

Hyväksytty tila on completed_exploratory_ml_ablation. Raportti: report.json, kummallekin jaksolle trades/daily/equity/summary.csv ja audit.json. Mahdollinen failed_not_validated ei ole tulos eikä oikeuta poistamaan päivää. Testattu synteettinen koko ketju (molempien jaksojen toisto, ei uudelleenkoulutusta), matalan pisteen osto vain positiivisella markkinatuotolla, signaalin ajoitus, vaihto/altistus nollapäivineen ja muuttuneen SPY-lähteen hylkäys. Kolme uutta testiä läpäistiin Python 3.12:lla pandas 2.2.3:ssa ja 3.0.0:ssa. Oikeaa 100 osakkeen aineistoa ei ollut tässä kehitysympäristössä, joten sen tuottotulosta ei raportoida.
