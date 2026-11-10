# Avaussignaalin etusija – erillinen tutkimuskoe

**Mitä tiedämme:** osakkeiden listajärjestys muutti sekä toteutuvia osakkeita että sektoreiden osuuksia. Elokuussa vaihtuva järjestys vähensi teknologiakauppoja 89:stä 52:een (100 kauppaa ilman SPY-porttia) mutta pienensi tuottoa. Lukitulla 20.10.–6.11. jaksolla vaihtuva järjestys kasvatti tuottoa, ja ilman porttia myös arvonlasku syveni. Kaikki neljä jälkimmäisen jakson päivätuottoerojen epävarmuusväliä sisälsivät nollan. Aiemmat kokeet eivät osoita, että ML-todennäköisyyden perusteella priorisointi parantaisi tuottoa.

**Yksi uusi hypoteesi:** samaan päivärajaan osuvien ostojen valintaa voisi parantaa antamalla vahvemman avaussignaalin osakkeille etusija kiinteän listan sijaan. Kello 09.35 New Yorkin aikaa järjestetään osakkeet juuri sulkeutuneen 09.30–09.35 kynttilän ml_prob-luvun mukaan laskevasti. Tasatilanteet ratkotaan alkuperäisellä järjestyksellä. Jos avausennuste puuttuu, osake pysyy mukana listan lopussa; myöhempää ennustetta ei käytetä sen tilalla. Päivän järjestys ei muutu seuraavilla kynttilöillä. Jokainen osto edellyttää edelleen **sen oman skannaushetken** alkuperäisiä signaali-, aika- ja riskiehtoja. Kokeessa ei säädetä mallia, positioiden kokoa, uutisia tai kynnysarvoja.

Ensisijainen vertailu on avaussignaalin järjestyksen ja kiinteän listan parittainen keskimääräinen päivätuottoero ilman SPY-porttia. Samat kaksi järjestystä raportoidaan myös portin kanssa. Molemmat historialliset jaksot pidetään mukana erikseen: elokuu sekä 20.10.–6.11.2026. Molempien alkuperäisiä tuloksia on aiemmin katsottu: tämä on **tutkimuskokeilu, ei riippumaton vahvistus**. Positiivinen tulos ei aktivoi muutoksia. Jos parempaa tuottoa havaitaan, tarkista myös arvonlasku, päivittäisten erojen epävarmuus, kauppamäärä ja keskittyminen. Eri aikojen tai järjestysten etsiminen jälkikäteen on uusi koe.

Ajo käyttää paikallisen elokuun välimuistia ja toisen jakson tallennettuja SIP-hintoja. Toisen jakson pisteet lasketaan jo jäädytetyllä mallilla. Se ei lue uusien seurantapäivien hintoja tai tuloksia. Ei koulutusta, verkkohakuja, .env-tiedostoja, toimeksiantoja tai kirjoituksia lukittuun seurantaan. Molempien jaksojen vertailuhaarojen koko kirjanpito ja yhteenveto on toistettava (absoluuttinen toleranssi 1e-6). Tarkistussummat varmistetaan ja kaikki haarat auditoidaan ennen hyväksyttyjä tuloksia. Alkuperäisten tuloskansioiden on oltava jäljellä. Jos jokin puuttuu, älä korvaa sitä eri aineistolla.

Aja siinä PowerShell-simulaatiokansiossa, jossa aiemmat paikalliset kokeet ovat:
```powershell
& {
    if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Vaara kehityshaara' }
    git pull --ff-only origin audit-hardening-v1
    if ($LASTEXITCODE -ne 0) { throw 'Git-paivitys epaonnistui' }
    $out = 'research_runs/avaussignaali-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    .\.venv\Scripts\python.exe -u -X utf8 tools\trial_open_priority.py --output-dir $out
}
```
Valmiin kokeen tila on `completed_exploratory_trial`. Molemmissa alakansioissa summary.csv, daily.csv, equity.csv ja trades.csv kuvaavat kaikkia neljää haaraa. audit.json sisältää päivittäiset prioriteetit ja avauspisteet, keskittymät ja parittaiset päivätuottoerot. Alkuperäinen symbol_order tarkoittaa aina paikkaa asetuksissa; vaihtoehdon scan_order osoittaa käytännön prioriteetin.

Testit: `python -m unittest test_open_priority`. Keinotekoisilla hinnoilla testataan avauskynttilän aikaleima, puuttuva ennuste, tasapisteet, tulevien ennusteiden riippumattomuus, päivärajan sitoma valinta, SPY-portti sekä kummankin jakson kirjanpidon toistaminen jäädytetystä aineistosta. Väärennetty vertailutulos hylätään, vaikka sille annettaisiin vastaava tiedostotiiviste. Aitoja 100 osakkeen paikallisia tuloksia tästä kokeesta ei ole saatu kehitysympäristössä.

10.11.2026: kaikki kolme uutta testiä läpäistiin Linuxissa sekä pandas 2.2.3:lla että 3.0.0:lla. Uusi testi on erillinen CI-vaihe; se ei muuta alkuperäistä neljän vaihtoehdon protokollaa.

## Pöytäkoneen simulaatio 10.11.2026

Käyttäjän toimittama PowerShell-tuloste: ohjelmaversio `532ed41`, ajokansio `research_runs/avaussignaali-20261110-150815`, tila `completed_exploratory_trial`. Luvut ovat tulosteen jaksoyhteenvetoja, eivät uutta ajoa kehitysympäristössä. Ero on avaussignaalin mukaan järjestetyn ja alkuperäisen vaihtoehdon jaksojen tuottojen erotus prosenttiyksikköinä.

| Jakso | SPY +0,10 % | Kauppoja / vaihtoehto | Alkuperäinen tuotto | Avausjärjestyksen tuotto | Ero | Suurin arvonlasku, alkuperäinen → avaus |
|---|---|---:|---:|---:|---:|---:|
| Elokuu (21 päivää) | ei | 100 | +0,4312 % | +0,3249 % | −0,1063 | −2,1597 % → −1,8552 % |
| Elokuu | kyllä | 20 | +0,2871 % | −0,4240 % | −0,7111 | −0,7518 % → −0,8023 % |
| 20.10.–6.11. (14 päivää) | ei | 69 | +0,5033 % | −0,6396 % | −1,1428 | −1,5861 % → −1,6626 % |
| 20.10.–6.11. | kyllä | 15 | +0,5233 % | −0,3867 % | −0,9100 | −0,4500 % → −0,9298 % |

Kauppamäärä pysyi samana kussakin vertailuparissa, mutta avausjärjestyksen tuotto ja profit factor olivat pienemmät kaikissa neljässä. Suurin arvonlasku pieneni vain elokuussa ilman porttia. Parittaisen päivätuottoeron estimaattia ja epävarmuusväliä ei ollut toimitetussa tulosteessa; taulukon erot eivät korvaa niitä. Jaksot oli katsottu aiemmin, eivätkä saman jakson kaksi SPY-vaihtoehtoa ole riippumattomia toistoja. Päätös: avaussignaalin mukaista järjestystä ei oteta käyttöön. Näillä jaksoilla ei valita uutta kynnystä tai järjestystä, eikä käynnissä olevaa neljän vaihtoehdon seurantaa muuteta.
