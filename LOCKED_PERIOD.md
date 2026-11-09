# Lukittu 20.10.–6.11.2026 vertailu

**Jakson valinta ennen tämän kokeen hintojen tai tuottojen tarkastamista:** aiemmin tutkittu syys–lokakuun kehitysjakso päättyi 19.10.2026. Vertailu alkaa seuraavasta istunnosta ja päättyy 9.11. ajankohtaa edeltävään viimeiseen kokonaiseen viikkoon. Lukitaan 14 istuntoa 20.–23.10., 26.–30.10. ja 2.–6.11. Jokaiselle odotetaan New Yorkin aikaa 09.30–15.55 alkavat 78 viiden minuutin kynttilää. Aikavyöhykkeen vaihto otetaan huomioon. Kaikki SPY:n odotetut kynttilät vaaditaan. Jakso on historiallinen: toteutus ei pysty varmistamaan, ettei kukaan ole aiemmin tarkastellut sitä.

Nykyinen järjestys ja daily_sha256_v1, kumpikin alkuperäisellä SPY ≥ 0,10 % -portilla ja ilman. Kaikki neljä raportoidaan, samat ennusteet, 100 osaketta, pääoma, kulut, riskirajat ja neutraalit uutiset. Ei parametrien hakua, paper- tai live-toimeksiantoja. Mitään vaihtoehtoa ei valita käyttöön tämän jakson perusteella.

- Ajo tarkistaa edellisen järjestyskokeen **reference**-hakemiston manifestin, asetukset, koulutuskoodin ja historian tiivisteet. Elokuun välimuisti sisältää elokuun ennusteita, ei mallipainoja. Siksi yksi vähämuistinen koulutus tarvitaan. Koulutus ja toteutuneiden labelien katkaisu ovat samat kuin elokuussa (2023-01-01 alkaen, katkaisu 2026-08-01 klo 00.00 NY). Koulutusmatriisin SHA-256:n pitää täsmätä. Kolmen ennalta valitun elokuun ennusteen pitää täsmätä jokaiselle osakkeelle absoluuttisella toleranssilla 1e-10; tämä ei ole kaikkien mahdollisten ennusteiden identtisyystodiste.
- Markkinadata luetaan vain data.alpaca.markets-palvelun historiallisella GET-kutsulla. Lähde on SIP, säätö raw, symbolien viitepäivä 2026-08-31. Nimetyn paikallisen tunnustiedoston sisältöä ei tulosteta. Raakojen OHLCV-arvojen CSV-kopiot tallentuvat uuden ajon sip_extension-hakemistoon. Etukäteen erikseen valmistetun, samaan hakuikkunaan rajatun CSV-hakemiston voi antaa --extension-dir-valitsimella. Tällaisen hakemiston alkuperää ei oleteta vahvistetuksi.
- 1.–19.10. käytetään vain piirteiden alustamiseen (vähintään 250 edeltävää havaintoa). Näiden päivien kauppoja tai tuottoja ei käytetä vertailussa. Testijakson ennusteisiin ei lasketa tulevia labeleita. Istuntokohtaiset puuttuvat osakehinnat raportoidaan eikä niitä täytetä. Avoimen position havaitsematon kynttilä hylkää vertailun. Kaikkien haarojen kirjanpidon ja tiivisteiden on täsmättävä; raportin tila ratkaisee, hyväksyttiinkö ajo. Koulutuksen ja replayn aikana verkkoyhteydet on estetty.
- Tulokset: summary.csv, daily.csv, equity.csv, trades.csv ja ledger_audit.json. report.json kirjaa protokollan, lähteiden tiivisteet, neljä parittaista päivätuottovertailua, SPY-portin sallimat päivät ja onnistuneen tilan completed_fixed_period_comparison. Neljä vertailua ovat SPY:n vaikutus kummassakin järjestyksessä ja järjestyksen vaikutus erikseen portilla/ilman. Pieni otos; ei kannattavuustodistetta. Puuttuvat tiedot, epäonnistunut haku tai eriävä koulutusmatriisi eivät tuota hyväksyttyä historiallista tulosta.

Aja pöytäkoneen nykyisessä kehityshaaran simulaatiokansiossa:

```powershell
& {
    if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Vaara kehityshaara' }
    git pull --ff-only origin audit-hardening-v1
    if ($LASTEXITCODE -ne 0) { throw 'Git-paivitys epaonnistui' }
    $src = "$env:USERPROFILE\Downloads\trade_bot_v5.2"
    $out = "research_runs/lukittu-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
    .\.venv\Scripts\python.exe -u -X utf8 tools\run_locked_period.py `
        --train-data "$src\data_sip" --env-file "$src\.env.v5" `
        --reference research_runs/jarjestys-20261109-130806/reference --output-dir $out
}
```

Jos haku pysähtyy, tarkista virheilmoitus (esim. HTTP 403 tarkoittaa, ettei kyseinen historiapyyntö onnistunut). Ei vaihtoa IEX-syötteeseen eikä osakkeiden poistoa onnistuneen tuloksen saamiseksi. Jos kaikki CSV:t jo haettiin, niitä voi käyttää uudessa tuloskansiossa --extension-dir-valitsimella; julkaise alkuperäisen pyynnön lähde- ja tiivistetiedot arviointia varten.

Testattu pienellä synteettisellä aineistolla: oikea koulutus, kaikki neljä haaraa, kaikki 14 istuntoa ja kesä-/talviajan raja, aiemman mallin ennusteankkurit, puuttuva SPY-kynttilä ja istunto, lämpenemishistoria, muuttunut koulutusdata, paikallisten tulosten ylikirjoitussuoja sekä historialukijan sivutus/HTTP-virheet ilman verkkoyhteyttä. [Offline-testien lokit](audit_evidence/2026-11-09-locked-period). Kehitysympäristön historiallisen ajon yrityksestä puuttui 93/101 vaaditusta koulutustiedostosta. Koko hintahistorian ajo ja oikea API-haku tältä jaksolta eivät ole osa tätä testinäyttöä.

## Pöytäkoneen historiallinen ajo

Käyttäjän lähettämä konsoliyhteenveto, `lukittu-20261109-151005`, tila `completed_fixed_period_comparison`. Aiemmin sovitut neljä vaihtoehtoa, 20.10.–6.11.2026, 14 istuntoa. Nämä luvut ovat simuloituja tuloksia; koko paikallista report.json-tiedostoa tai päivä- ja kauppalokia ei ole tässä tarkastettu itsenäisesti.

| Järjestys | SPY-portti | Tuotto (%) | Kaupat | Voitto-osuus (%) | Profit factor | Suurin arvonlasku (%) | Keskimääräinen P&L |
|---|---|---:|---:|---:|---:|---:|---:|
| Alkuperäinen | Ei | 0,503251 | 69 | 46,376812 | 1,107118 | −1,586056 | 7,293494 |
| Alkuperäinen | ≥ +0,10 % | 0,523348 | 15 | 53,333333 | 1,698130 | −0,449953 | 34,889880 |
| Päivittäin vaihtuva | Ei | 1,202146 | 69 | 49,275362 | 1,267925 | −2,071467 | 17,422413 |
| Päivittäin vaihtuva | ≥ +0,10 % | 0,975836 | 15 | 66,666667 | 2,467244 | −0,275011 | 65,055731 |

Samalla SPY-asetuksella päivittäin vaihtuvan järjestyksen tuottoero alkuperäiseen oli ilman porttia +0,698895 prosenttiyksikköä ja portilla +0,452488 prosenttiyksikköä. Ilman porttia suurin arvonlasku syveni, portilla pieneni. Elokuussa järjestyksen tuottoerot olivat negatiivisia molemmilla SPY-asetuksilla. Portti vähensi kauppoja 69:stä 15:een; sen tuottovaikutus oli eri suuntainen kahdella järjestyksellä. Portin sallimien päivien määrä ja parittaisten erojen epävarmuus luetaan paikallisesta raportista eikä niitä päätellä tästä taulukosta. Ei parametrien vaihtoa, uutta sovitusta tai tuotantopäätöstä.
