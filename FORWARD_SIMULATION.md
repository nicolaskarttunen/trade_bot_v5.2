# Neljän vertailuversion tulevien päivien seuranta, 10.11.–8.12.2026

Kiinteä kohortti: seuraavat 20 suunniteltua Yhdysvaltain osakemarkkinoiden istuntoa 9.11. päätöksen jälkeen. Päivälista on `tools/watch_forward.py`-tiedostossa. Kiitospäivä 26.11. jätetään pois ja 27.11. sulku on 13.00 New Yorkin aikaa. Jokainen SPY:n säännöllisen istunnon kynttilä vaaditaan myös lyhyenä päivänä. Istuntoja ei korvata jälkikäteen muilla päivillä.

Alkuperäinen ja daily_sha256_v1-järjestys, molemmat ilman SPY-porttia ja kiinteällä +0,10 % portilla. Jokaisella haaralla on oma 100 000:n alkupääoma, kassankertymä ja samat riskit, kulut ja neutraalit uutiset kuin historiallisessa kokeessa. Raportoitavat vertailut ovat samat neljä parittaista päivätuottoeroa kuin lukitulla edellisellä jaksolla. Kaikki päivät ja kaikki neljä haaraa pidetään mukana. Ajossa ei valita voittajaa, säädetä rajoja tai tehdä toimeksiantoja. Vertailu kuvaa päätösten jälkeistä **päivän sulkeuduttua tehtyä OHLCV-simulaatiota**. Datatoimittajan korjaukset ennen tallennusta ja todellinen toteutushinta voivat poiketa reaaliaikaisesta kaupasta.

## Käynnistys omalla Windows-koneella

Aloita ennen tiistain 10.11. avausta (16.30 Suomen aikaa). Lukitus tapahtuu ennen mallin rakentamista. Aiemmin rekisteröidyn ajon saa käynnistää uudelleen. Uuden jälkikäteen aloitetun seurannan nimeäminen samaksi etukäteiskokeeksi estetään.

```powershell
& {
    Set-Location "$env:USERPROFILE\Downloads\tradebot-simulaatio-20261026-225346" -ErrorAction Stop
    if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Vaara kehityshaara' }
    git pull --ff-only origin audit-hardening-v1
    if ($LASTEXITCODE -ne 0) { throw 'Git-paivitys epaonnistui' }
    powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File .\tools\start_forward.ps1
}
```

Käynnistin olettaa saman työaseman data_sip-, .env.v5- ja elokuun reference-kansion kuin edellinen koe. Mallipainoja ei tallennettu siinä ajossa. Nyt tehdään kerran sama elokuussa todennettu koulutus, tarkistetaan vanha matriisitiiviste ja ennusteankkurit ja tallennetaan numeeriset painot JSON-muodossa ainoastaan hakemistoon `research_runs/forward-20261110`. JSON-lataus ei suorita koodia. Alkuperäisiä malleja, avaimia ja asetustiedostoja ei ylikirjoiteta. Mallin, lähdekoodin, versioiden tai asetusten poikkeama pysäyttää seurannan. Alustukseen voi mennä tunteja; jätä ikkuna auki, kunnes näet rekisteröidyn tehtävän ja tilan. Jos Windowsin paikallinen käytäntö estää rekisteröinnin, ohjelma ilmoittaa virheen eikä väitä ajastuksen toimivan.

Windowsin tehtävä **TradeBot-Simulation-Forward-20261110** käynnistyy nykyisen käyttäjän oikeuksilla päivittäin 21.45 UTC (23.45 Suomen aikaa). Tehtävä on aikarajattu, ei käynnistä varsinaista bottia eikä vaadi tallennettua Windows-salasanaa. Windows-käyttäjän pitää olla kirjautuneena ja koneen käytettävissä. StartWhenAvailable pyytää Windowsia suorittamaan väliin jääneen käynnistyksen myöhemmin; mitään etäkoneen ajastusta ei synny, ennen kuin paikallinen rekisteröinti onnistuu.

Päivä tulee käsittelyyn aikaisintaan 30 min pörssin sulkeutumisen jälkeen. Haku on vain GET SIP/raw/5Min, symboliviitepäivä 2026-08-31. Jokaiselle osakkeelle haetaan samanaikaisesti käytettävissä oleva historiallinen ikkuna, jonka alku on 14 kalenteripäivää ennen kohdepäivän alkua ja loppu ennen istunnon sulkua. Haku päättyy viimeistään viimeiseen sulkeutuneeseen kynttilään. Vähintään 250 aiempaa havaintoa vaaditaan. Yö-/esikauppahistoria voi vaikuttaa piirteisiin kuten aiemmin. Ei tulevien päivien hakuja. Jokaisen päivän syöte jäädytetään omaan tiivistettyyn tiedostoon ennen tuloslaskentaa. Jos ensimmäinen haku päättyy vasta myöhempänä NY-päivänä, päivä näkyy late_sessions-listassa. Sulun jälkeistä eikä jälkikäteen kerättyä bar-dataa tule kuvata reaaliaikaisina havaintoina.

## Tarkistus ja toipuminen

```powershell
$state = '.\research_runs\forward-20261110'
Get-Content "$state\last_attempt.json"
Get-Content "$state\status.json"
if (Test-Path "$state\summary.csv") { Import-Csv "$state\summary.csv" | Format-Table -AutoSize }
Get-ScheduledTaskInfo -TaskName TradeBot-Simulation-Forward-20261110
```

Viimeisin hyväksytty status on waiting_first_close, collecting tai complete. last_attempt.json kertoo myös viimeksi epäonnistuneesta yrityksestä; edellinen summary voi silloin olla vanha. Päiväkansiot sisältävät kaikille haaroille trade-, daily-, equity- ja summary-taulukot sekä audit- ja manifestitiedostot. Päivän tiedot hyväksytään yhdessä, kun kaikki neljä laskelmaa täsmäävät. Päivän epäonnistuminen ei salli sen yli hyppäämistä. Keskeytyksen jälkeen sama asennuskomento tai alla oleva päivitys jatkaa; hyväksyttyjä päiviä tai jäädytettyjä syötteitä ei lasketa uudelleen muokatulla datalla. Älä poista aukollisia päiviä tai muuta asetuksia kesken seurannan.

```powershell
.\.venv\Scripts\python.exe -u -X utf8 tools\watch_forward.py update --state research_runs\forward-20261110 --env-file "$env:USERPROFILE\Downloads\trade_bot_v5.2\.env.v5"
```

Lopulliset neljä parittaista vertailua kirjataan status.json-tiedostoon vain 20 hyväksytyn päivän jälkeen. Tämä ei ole kannattavuus- tai käyttöönottolupa. Tämän tietyn ajastuksen voi poistaa Windowsista komennolla `Unregister-ScheduledTask -TaskName TradeBot-Simulation-Forward-20261110`; simulaation tulokset jäävät talteen.

Synteettiset testit kattavat numeerisen mallin tallennuksen ja samat ennusteet uudelleenladattuna, 20 session kalenterin, lyhyen päivän, kellon ohjauksen, hylätyn päivän, syötteen uudelleenkäytön, pitkän offline-jakson ja myöhäisten päivien merkinnän, kassankertymän, loppukoosteen ja tiivistevirheet. Todellisia tulevia hintoja tai tuottoja ei vielä ole. Windows-CI muodostaa tehtävämäärittelyn asentamatta käyttäjälle tehtävää. [Testitulokset](audit_evidence/2026-11-09-forward).
