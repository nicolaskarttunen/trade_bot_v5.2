# Elokuun simulaation vähämuistinen toteutus

Vain `audit-hardening-v1`. Ei Oracle-palvelimen, kaupankäyntiparametrien,
osakejoukon, aikarajojen tai tuotantomallien muutoksia.

## Miksi tämä lisättiin?

Käyttäjän esitarkistus löysi enintään 10 018 854 raakakoulutusriviä. Yksi
17 piirteen float64-matriisi on noin 1,27 GiB, mutta aiempi toteutus säilytti
kaikkien osakkeiden koulutuskehykset ja teki lisäksi yhdistely-, skaalaus- ja
bootstrap-kopioita. Käyttäjän 15,8 GiB koneella vapaana oli ensin 3,4 GiB,
ohjelmien sulkemisen jälkeen 8,0 GiB. Aiempi 8–10 GiB:n tavoite oli karkea
käyttövara, ei mitattu RAM-huippu tai takuu alkuperäisen ajon onnistumisesta.

## Mitä muuttui?

`--low-memory` on erillinen valitsin. Oletusajo ja aiempi RAM-toteutus
säilyvät vertailua varten. `config.json` pysyy ennallaan.

- Piirteet ja labelit tehdään edelleen yhteisillä funktioilla yhdelle
  osakkeelle kerrallaan. Koulutusrivit kirjoitetaan symbolikohtaisiin
  väliaikaistiedostoihin. Koko osakehistoriaa käsitellään yhä yksi symboli
  kerrallaan RAMissa; tätä osaa ei ole muutettu yleiseksi CSV-virtalukijaksi.
- Levylle tehtävä yhdistely tuottaa saman vakaan aikaleima- ja symbolijärjestyksen
  kuin RAM-ajurin concat + stable sort. Jokaisesta lähdetiedostosta pidetään
  yhdistelyn aikana vain 1024 riviä muistissa.
- Koulutus lukee enintään 32 768 rivin lohkoja. **Painot päivitetään vain kerran
  koko aineiston läpikäynnin jälkeen**, kuten alkuperäisessä full-batch-
  gradienttilaskennassa. Tämä ei ole mini-batch-/SGD-koulutuksen vaihto.
- Koko koulutushistoria säilyy. LogisticGD: 500 kierrosta. RandomLinearEnsemble:
  samat 25 mallia, siemen 42, sarakevalinnat ja 250 kierrosta per malli.
  Bootstrap-arvonta on täsmälleen sama RNG-kutsu; toistuvat rivit esitetään
  lukumääräpainoina kopioiden sijasta. Bootstrapin nollapainoisia rivejä ei
  tarvitse kirjoittaa kyseisen mallin skaalausvälimuistiin. Otoskokoa ei pienennetä.
- Standardoinnin keskiarvot/varianssit ja centroidin margin_scale lasketaan
  koulutusaineistosta erissä. Ennusteet tekee edelleen sama alkuperäinen
  malliluokka. Mallien määrittely ja bootstrap-jäsenen asetukset tulevat
  yhteisistä tehtaista; RAM-koulutuksen toiminta ei muutu.
- Replay käyttää samaa moottoria yhden session kerrallaan ja siirtää edellisen
  session todellisen loppukassan seuraavan alkuun. Päivän lopussa positiot ovat
  kiinni alkuperäisen tiukan EOD-säännön mukaan. Näin kaikkien päivien
  pandas-riviolioita ei tarvitse pitää moottorin sanakirjoissa samanaikaisesti.
- CLI rajoittaa OpenBLAS-/OMP-työntekijät yhteen vain omassa prosessissaan.
  Se ei muuta järjestelmän asetuksia tai sulje käyttäjän ohjelmia.

Liukulukusummien järjestys muuttuu, joten bittitarkkaa identtisyyttä ei luvata.
Synteettisissä testeissä ennuste-ero alitti 1e-11 absoluuttisen toleranssin ja
kokonaisen elokuun kokeen kaupat/tulokset täsmäsivät testitoleransseissa.
Täsmälleen päätösrajan lähellä pyöristys voi silti vaikuttaa signaaliin.
Kynnysarvoja ei säädetä tämän peittämiseksi.

## Levyn käyttö ja virheet

Ennen raskasta ajoa tarkistetaan vapaa levytila: kolme raakakoulutusrivimäärän
mukaista binääritiedostoa + 1 GiB käyttövara, tässä aineistossa noin 5,3 GiB.
Tämä ei ole lupaus koko koneen vapaasta RAMista tai muiden prosessien käytöstä.

Välitiedostot ovat vain uuden tuloskansion `training_cache`-hakemistossa.
Symboliosat poistetaan onnistuneen yhdistelyn jälkeen. Skaalausvälimuisti
poistetaan kunkin mallin jälkeen, myös käsitellyn keskeytyksen yhteydessä.
Yhdistetty koulutustiedosto poistetaan onnistuneen koulutuksen jälkeen ja sen
tiiviste säilytetään raportissa. Tuntemattomia tiedostoja ei poisteta
rekursiivisesti; jos niitä jää kansioon, asia kirjataan raporttiin.

Keskeytettyä ajoa ei jatketa automaattisesti vanhoista välimuistitiedostoista.
Uusinta tarvitsee uuden tuloskansion. Data-aukot, puuttuvat sessiot, muuttuneet
lähteet ja epäonnistunut haara hylkäävät edelleen koko tuottovertailun.
Välittäjäyhteyksiä tai avaintiedostoja ei avata.

## Oikeasti testattu

Kymmenen uutta regressiotestiä:

1. Ulkoinen yhdistely säilyttää kaikki arvot ja saman aikaleiman symbolijärjestyksen.
2. Koko kiinteän ensemblen ennusteet, skaalat, bootstrap-sarakkeet ja painot
   vastaavat RAM-versiota synteettisellä aineistolla; kierrosmäärät pysyvät samoina.
3. Lukumääräpainotetut täydet gradienttipäivitykset vastaavat monistettua
   bootstrap-otosta myös hyvin pienillä lohkoilla.
4. Virheelliset arvot, yhden luokan aineisto ja katkennut binääritiedosto hylätään.
5. Keskeytys siivoaa vain oman skaalausvälimuistin, ei koulutusaineistoa
   tai ennalta olemassa olevaa tiedostoa.
6. Sessioittainen replay vastaa koko jakson replayta kassankertymineen ja
   nollakauppapäivineen molemmilla SPY-asetuksilla.
7. Oikea vähämuistinen koulutus + elokuun replay vastaa RAM-ajoa kahdella
   synteettisellä osakkeella ja 21 sessiolla, alkuperäiset pysyvät ennallaan.
8. Avoimen position puuttuva kynttilä estää myös vähämuistisen tuottotuloksen.
9. Riittämätön levytila estää koulutuksen aloittamisen.
10. Välimuistikansion tuntematon tiedosto säilyy ja kansion jääminen raportoidaan.

Koko offline-sarjassa on 126 testiä. Tulokset ja lähdekoodin tiivisteet:
`audit_evidence/2026-10-28-low-memory/`. GitHub Actions ajaa Windows/Linux ×
pandas 2.2.3/3.0.0 -matriisin. Lopullinen CI-tila tarkistetaan commitin ajosta.

### Suuren aineiston muistimittauksen rajat

`tools/measure_low_memory.py` luo 10 018 854 keinotekoista riviä ja 100
symboliosaa. Se testaa yhdistelyn, skaalausten muodostamisen sekä yhden
täyden gradienttipäivityksen päämallille ja yhdelle bootstrap-mallille.
Prosessille asetetaan Linuxissa 2 GiB virtuaaliosoiteavaruuden raja.
Tulokset tallennetaan `memory_smoke.json`:iin.

Tämä EI aja 500 + 25×250 kierroksen koko ensemblea, suuren aineiston
piirteiden muodostusta tai koko 100 osakkeen replayta. Mitattu RSS ei sisällä
käyttöjärjestelmän tiedostovälimuistia eikä ole Windowsin huippumuistin mittaus.
Mittauksesta ei päätellä koko historiallisen ajon kestoa tai kannattavuutta.

## Windows-ajo

```powershell
git pull --ff-only origin audit-hardening-v1
if ($LASTEXITCODE -ne 0) { throw 'Git update failed' }

.\.venv\Scripts\python.exe -u -X utf8 tools\run_august_simulation.py `
    --data-dir "$env:USERPROFILE\Downloads\trade_bot_v5.2\data_sip" `
    --output-dir research_runs\elokuu-simulaatio-lowmem-1 `
    --run --low-memory
```

Ajo tarkistaa lähteet, tekee piirteet, kouluttaa ja simuloi. Edistyminen
tulostetaan mallikohtaisesti 25 koulutuskierroksen välein. Koko koulutuksen
kestoa käyttäjän koneella ei ole mitattu. Pidä kone hereillä ja anna ajon
valmistua. Tulokset pysyvät aiemman elokuun kehitystestiprotokollan mukaisina;
mitään riippumatonta kannattavuusnäyttöä tai käyttöönottoa ei väitetä.

Lopullisen Linux-muistikoekayton mitattu RSS-huippu oli 243.0 MiB.
Mittauksen rajat ovat edella; konekohtainen kokonaisajo on viela tekematta.
Tarkka raportti: `audit_evidence/2026-10-28-low-memory/memory_smoke.json`.
