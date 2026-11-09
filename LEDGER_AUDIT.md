# Elokuun simulaation kauppalokin tarkistus

Aiemmin valmistuneen ajon voi tarkistaa ilman hintahistoriaa, koulutusta tai välittäjätunnuksia. `tools/audit_simulation_run.py` lukee tulokset, vertaa tiedostojen SHA-256-tiivisteitä ajon manifestiin ja kirjoittaa raportin uuteen kansioon. Käytä kyseisen ajon omia tulostiedostoja yhdessä. Alkuperäisten tulosten päälle tai sisään ei kirjoiteta.

```powershell
$src = '.\research_runs\elokuu-lowmem-20261028-141359'
$out = '.\research_runs\lokiaudit-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.\.venv\Scripts\python.exe -X utf8 tools\audit_simulation_run.py --run-dir $src --output-dir $out
```

`passed_ledger_checks` tarkoittaa, että molempien haarojen tulokset täsmäsivät kirjattuihin arvoihin. `failed_ledger_checks` palauttaa virhekoodin; löydökset ovat `audit_report.json`-tiedoston `error`- tai `arms.*.issues`-kentässä. Tarkistustyökalu ei avaa alkuperäisiä hintoja eikä todenna välittäjän toteutushintoja tai voiton toistettavuutta.

- Jokaisen kaupan P&L, suunniteltu stop-tappio, kirjattu altistus, osakemäärä, hinta- ja aikarajat tarkistetaan tallennetuilla asetuksilla. Avausgap voi tuottaa suunniteltua suuremman tappion; sitä ei hylätä pelkästään tästä syystä.
- Kaupat täsmäytetään kaikkien 21 session päivätuloksiin, viiden minuutin pääomakäyrään ja yhteenvedon kuuteen mittariin. Myös nollakauppapäivät pidetään mukana. Samanaikaisten positioiden lukumäärä perustuu kirjattuihin aikaväleihin, ei tick-tason hintapolkuun.
- Raportissa ovat kauppamäärät, osuudet ja P&L osakkeittain ja sektoreittain, ensimmäisten 20 osakkeen osuus kaupoista sekä parhaan päivän ja muiden päivien tulokset. Osuudet kuvaavat kauppamääriä, eivät sijoitettua pääomaa.
- Uusi elokuun ajo tarkistaa molemmat haarat ennen vertailun hyväksymistä ja tallentaa `ledger_audit.json`-tiedoston. Se kirjoittaa `decisions.jsonl`-tiedostoon yhden syyn per osake käsitellyllä skannaushetkellä. Syitä ovat esimerkiksi `entered`, `threshold_not_met`, `open_position_limit` ja `size_below_one`. Päivärajan täytyttyä loppulistan syy on `daily_trade_limit_not_evaluated`: signaalin kelpoisuutta ei tutkittu. SPY-portti ja päivätappioraja erottavat vastaavasti tarkistamattomat osakkeet. Epäonnistuneen ajon syyloki voi olla vajaa.
- Vanhaan ajoon ei päätellä puuttuvia hylkäyksiä. Tiedostotiivisteet kertovat muutoksista manifestiin nähden, eivät aineiston alkuperästä. Tämä muutos ei muuta valintajärjestystä, riskiasetuksia tai SPY-rajaa.

## Näyttö

Käyttäjän 28.10. raportoimassa ajossa teknologiakauppoja oli 89/100 ja SPY-portilla 20/20. Ensimmäisten 20 osakkeen kauppamäärät olivat vastaavasti 89 ja 20. Tämä havainto perustuu käyttäjän PowerShell-tulosteeseen; kyseisen koneen kaikkia kaupparivejä ei ole tarkistettu täällä.

Linux/Python 3.12: 137/137 testiä onnistui erikseen pandas 2.2.3:lla ja 3.0.0:lla. Lokit ja lähdekooditiivisteet: [testitulokset](audit_evidence/2026-11-09-ledger). Testit sisältävät väärän P&L:n, riskirajojen ja tiivisteiden poikkeamat, nollakauppapäivät, gappitappion, muuttumattoman vanhan ajon, syylokin virheen sekä ajon hylkäämisen ennen tuottoraporttia. Synteettisessä parivertailussa lokitus päällä/pois tuotti samat kaupat, kassavirran ja pääomakäyrän. Voimassa oleva Windows/Linux-CI ajaa saman sarjan molemmilla pandas-versioilla.
