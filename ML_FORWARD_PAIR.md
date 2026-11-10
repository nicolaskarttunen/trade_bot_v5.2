# ML-suodatin vastaan pelkkä markkinaehto: etukäteen lukittava simulaatio

Uusi havaintojakso: **11.11.–9.12.2026**, 20 Yhdysvaltain pörssipäivää. Rekisteröinti tehdään ennen 11.11. klo 09.30 New Yorkin aikaa (16.30 Suomessa). Ajastus ei siirrä rekisteröintirajaa. 10.11. ei ole koepäivä. Kiitospäivä 26.11. jää pois, 27.11. sulku on 13.00 ET ([NYSE:n kalenteri](https://www.nyse.com/trade/hours-calendars)).

Ensisijainen kysymys: parantaako jäädytetyn mallin osakekohtainen kelpoisuusehto keskimääräistä päivätuottoa mallinnettujen kulujen jälkeen? Alkuperäisellä osakejärjestyksellä molemmat vaativat positiivisen lyhyen SPY-tuoton ja muut samat aika-, data- ja riskiehdot. original_ml edellyttää lisäksi ennustetta vähintään 0,55. market_only hyväksyy myös pienemmät pisteet ja käyttää täsmälleen samoja piirrerivejä. Kummassakaan ei ole SPY:n erillistä avausporttia; uutiset ovat neutraaleja. Aloitus erillisillä 100 000 USD:n kassoilla ilman aiempien päivien voittoja. Ei koulutusta tai toimeksiantoja.

Rekisteriin lukitaan päivät, koodi ja ympäristöversiot, mallin ja sen alkuperäisen rekisterin tiivisteet, asetukset, vertailun suunta ja mittarit. 19 ensimmäistä päivää luetaan muuttamatta nykyisen forward-20261110-seurannan sulun jälkeen tallentamia hintapaketteja. Aiemman kokeen kauppoja tai tuottoja ei käytetä. Vain 9.12. hinnat (sama SIP/raw/5Min, 14 kalenteripäivän lämmittely ja sama asof) haetaan vanhan kerääjän rajatulla historiallisen datan GET-toiminnolla, erilliseen kansioon. Aikaisintaan 30 minuuttia sulun jälkeen. Puuttuva päivä, muuttunut lukittu lähde, puuttuva vertailuhinta tai avoimen position havaintoaukko estää kokonaisen tuloksen. Myöhemmin haetut päivät merkitään erikseen report.json-tiedostoon. Ei väliarvostelua, päivien korvaamista, aikaisen lopetuksen sääntöä tai rajan uudelleenvalintaa.

Lopussa original_ml miinus market_only mitataan parittaisista päivätuotoista (bp/päivä, myös nollapäivät, 5 päivän lohkot, 2 000 toistoa, siemen 20261023). Lisäksi tuotto, arvonlasku, kustannusstressi +3 bp ostolle ja myynnille, aika- ja nimellisarvopainotettu arvio altistuksesta, vaihto ja tulos ilman parasta päivää. Jos väli sisältää nollan, näyttö edusta jää epäselväksi. Myös uudella jaksolla 20 päivää on pieni otos ja toteutukset ovat OHLCV-simulaatiota, eivät välittäjän hintoja. Pienempi altistus voi selittää osan erosta. Positiivinen tulos ei muuta tuotantoa.

## Asennus omalla Windows-koneella ennen ensimmäistä avausta

Nykyisessä audit-hardening-v1-kansiossa päivitä kehityshaara ja aja PowerShellissä:

```powershell
& {
    if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Vaara kehityshaara' }
    git pull --ff-only origin audit-hardening-v1
    if ($LASTEXITCODE -ne 0) { throw 'Git-paivitys epaonnistui' }
    powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File .\tools\start_ml_pair.ps1
}
```

Onnistuminen: *Registered ML pair*, *Scheduled task: TradeBot-Simulation-MLPair-20261111* ja seuraava ajo. Asennus ei lataa päivän hintoja. Vanha kerääjä jatkaa 19 yhteisen päivän hintapakettien tekemistä; Windows-käyttäjän pitää olla kirjautuneena ja koneen hereillä ilta-ajoille. Uusi tehtävä käynnistää loppulaskennan **9.12. klo 23.45 Suomen aikaa**. Windows saa yrittää uudelleen seuraavina iltoina 17.12. asti. Valmista tulosta ei lasketa uudelleen. Laskennan jälkeen:

```powershell
$r = '.\research_runs\ml-pair-20261111\result'
Import-Csv "$r\summary.csv" | Format-Table -AutoSize
(Get-Content "$r\report.json" -Raw | ConvertFrom-Json) | Select-Object status,late_sessions,ml_minus_market_only | ConvertTo-Json -Depth 4
```

Jos raporttia ei synny, katso Get-ScheduledTaskInfo -TaskName TradeBot-Simulation-MLPair-20261111 ja kokeile loppuajon jälkeen käsin nähdäksesi täsmällisen virheen. Sama rekisteri ja talletettu viimeisen päivän paketti käytetään uudelleen:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 tools\ml_forward_pair.py update --state research_runs\ml-pair-20261111 --source research_runs\forward-20261110 --env-file "$env:USERPROFILE\Downloads\trade_bot_v5.2\.env.v5"
```

## Mitä testattiin

Linux/Python 3.12: ML-parin, edellisen ML-analyysin ja alkuperäisen forward-kerääjän sarjat (8 testiä yhteensä kummallakin pandas-versiolla, 2.2.3 ja 3.0.0). Keinotekoisella aineistolla 20 päivää päästä päähän, kaksi kassapolkua, ero oikeaan suuntaan, rekisteröinnin määräaika, lyhyt päivä, myöhäinen haku, ennenaikaisen loppuajon esto, toistettava ajo, muuttuneen hintapaketin ja tuloksen hylkäys sekä vanhan lähdekansion muuttumattomuus. Uuden Windows-tehtävän määrittely testataan CI:ssä asentamatta tehtävää. Tässä ei ole rekisteröity tehtävää käyttäjän koneelle eikä saatu tuottotulosta tulevilta päiviltä.
