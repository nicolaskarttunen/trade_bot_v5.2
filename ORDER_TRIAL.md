# Yksi erillinen järjestyskoe elokuun simulaatioon

Koe vastaa yhteen kysymykseen: muuttuuko valinta, riski tai tulos, kun listan alkupään osakkeet eivät saa jatkuvasti etusijaa? Muuttuja on pelkkä skannausjärjestys. Vertailuna on nykyinen järjestys. Kokeessa osakkeet järjestetään päivittäin SHA-256-arvolla merkkijonosta `daily_sha256_v1|YYYY-MM-DD|SYMBOL`. Kauppapäivä ja symbolit ovat tiedossa ennen avausta. Järjestystä ei valita signaalien, sektorin tai tuottojen perusteella. Sama päivittäinen järjestys pätee päivän kaikkiin skannauksiin. Järjestys lukitaan raporttiin ennen kokeellisen tuloksen laskemista.

Molemmat järjestykset ajetaan sekä ilman SPY-porttia että alkuperäisellä +0,10 % portilla, samoilla kiinteillä ennusteilla, hinnoilla, kustannuksilla ja riskirajoilla. Kumpaakaan SPY-versiota ei poisteta raportista. Protokolla on määritetty sen jälkeen, kun elokuun nykyisen järjestyksen tulos nähtiin: tämä on kehityskoe. Strategian asetuksia ei valita tuotantoon elokuun voittajan perusteella.

## Aja pöytäkoneen aiemmassa simulaatiokansiossa

Tarvitaan jo käytössä oleva Python 3.12 -virtuaaliympäristö ja ensin ajettu elokuun vertailu. Ensimmäinen komento päivittää vain kehityshaaran. Varsinainen koe käyttää paikallisia tiedostoja eikä tarvitse API-avaimia.

```powershell
& {
    if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Tarkista kehityshaara' }
    git pull --ff-only origin audit-hardening-v1
    if ($LASTEXITCODE -ne 0) { throw 'Git-paivitys epaonnistui' }
    $out = "research_runs/jarjestys-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
    .\.venv\Scripts\python.exe -u -X utf8 tools\run_order_trial.py `
        --previous-run research_runs/elokuu-lowmem-20261028-141359 `
        --data-dir "$env:USERPROFILE\Downloads\trade_bot_v5.2\data_sip" `
        --output-dir $out
}
```

1. Vanhan ajon manifesti, tiivisteet, kirjanpito ja riskirajat tarkistetaan ennen koulutusta. Virhe pysäyttää kokeen. Alkuperäistä kansiota ei muokata.
2. Jos alkuperäisessä ajossa ei ole ennusteita, samoista lähteistä koulutetaan yksi vähämuistinen vertailumalli. Tämä voi viedä tunteja. Malleja ei kouluteta erikseen eri järjestyksille. Uudessa `reference`-kansiossa ovat myös jäädytetyt elokuun ennusteet ja hinnat (`replay_inputs_v1.zip`, CSV/JSON).
3. Tarkistetaan alkuperäisen järjestyksen koko kauppaloki, päivät, pääomakäyrä ja yhteenveto. Rahamäärien ja muiden numeeristen kenttien vertailun absoluuttinen toleranssi on 1e-6; rivien ja luokittelujen pitää täsmätä. Myös välimuistista uudelleen ajettu vertailu täsmäytetään. Ero pysäyttää kokeen ennen järjestysvaihtoehtoa.
4. Uuden järjestyksen tulokset ovat `changed_order`-kansiossa. `symbol_order` on aina paikka **alkuperäisessä** listassa; `scan_order` kertoo toteutuneen käsittelyjärjestyksen. Molempien SPY-haarojen keskittymät ja tarkistukset ovat `ledger_audit.json`-tiedostossa. Mukana ovat myös nollakauppapäivät. `order_report.json` sisältää päivittäisen järjestyksen, tiedostotiivisteet sekä parittaisen päivätuottoeron kummallekin SPY-haaralle (muutettu miinus alkuperäinen).
5. Vain tila `completed_order_experiment_development_only` kertoo kokeen valmistuneen. Virheen jälkeen mahdollisesti syntyneet osatulokset eivät ole hyväksytty vertailu. Välimuistin myöhempää kokeilua varten käytä `--previous-run <uuden ajon reference-kansio>` ja uutta tuloskansiota. Silloin `--data-dir` voidaan jättää pois eikä mallia kouluteta uudelleen.

Koetulokset eivät muuta `config.json`-tiedostoa, mallipainoja, Oracle-palvelinta tai toimeksiantoja. Uutiset ovat edelleen neutraaleja. Tämä koe ei ratkaise aiemmin dokumentoituja välittäjäintegraation käyttöönottoesteitä.

## Testinäyttö

Synteettiset testit tarkistavat lukitun järjestyksen, listan alkuperäisten paikkojen säilymisen, päivärajan sitoman ensimmäisen valinnan sekä tarkkojen liukulukujen säilymisen CSV-välimuistissa. Integraatiotesti ajaa koko elokuun kahdella keinotekoisella osakkeella: vanha ajomuoto, yksi uusi koulutus, uusi järjestys, seuraava ajo ilman koulutusta sekä hylkäys lähteen tai vertailutuloksen poiketessa. Koko testisarjan ajolokit: [Linux, molemmat pandas-versiot](audit_evidence/2026-11-09-order-trial). Paikallisen historiallisen ajon tulokset on kirjattu alle erikseen.

9.11.2026: koko offline-sarja läpäisi 141/141 testiä Linuxissa sekä pandas 2.2.3:lla että 3.0.0:lla, ei ohitettuja testejä. Tämä ei ole käyttäjän historiallinen tuottotulos.

## Paikallinen historiallinen ajo 9.11.2026

Lähde: käyttäjän PowerShell-tuloste, ajo `jarjestys-20261109-130806`, tila `completed_order_experiment_development_only`. Jakso: elokuun 2026 kaikki 21 istuntoa, 100 osakkeen valikoima. Alkuperäisen järjestyksen vertailu toistui ajurin toleransseissa. Alla olevat luvut perustuvat ajurin yhteenvetoon ja kauppalokin sektorilaskentaan; koko paikallista aineistoa ei kopioitu repositorioon.

| SPY-portti | Järjestys | Kauppoja | Teknologiakauppoja | Alkuperäisen listan 20 ensimmäistä | Tuotto | Suurin arvonlasku |
|---|---|---:|---:|---:|---:|---:|
| Ei | Alkuperäinen | 100 | 89 % | 89 % | +0,431 % | −2,160 % |
| Ei | Päivittäin vaihtuva | 100 | 52 % | 56 % | +0,194 % | −1,297 % |
| +0,10 % | Alkuperäinen | 20 | 100 % | 100 % | +0,287 % | −0,752 % |
| +0,10 % | Päivittäin vaihtuva | 20 | 40 % | 40 % | +0,144 % | −0,433 % |

Muuttunut järjestys vähensi listan alkupään ja teknologiasektorin osuutta toteutuneiden kauppojen lukumäärästä. Osuudet eivät mittaa pääomapainoja. Ilman porttia 52/100 kauppaa kohdistui edelleen teknologiasektoriin. SPY-portti salli kaupankäynnin vain neljänä päivänä; sen haaran 20 kauppaa eivät ole 20 riippumatonta päivähavaintoa. Molemmissa haaroissa myös arvonlasku ja tuotto pienenivät. Järjestys säilyy kokeellisena, oletusasetuksia ei vaihdeta. Mahdollinen seuraava vahvistus tehdään ennalta lukitulla, aiemmin tarkastelemattomalla jaksolla ilman elokuuhun perustuvaa säätöä.

Seuraavan ennalta lukitun historiallisen jakson [protokolla ja ajokomento](LOCKED_PERIOD.md).
