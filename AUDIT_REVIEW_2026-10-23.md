# Auditoinnin jatko 23.10.2026 — vain kehityshaara

Kohde: `nicolaskarttunen/trade_bot_v5.2`, `audit-hardening-v1`.
Tarkastuksen lähtöcommit: `f92c46af5c5ee677505b542fe48035d9a82e3047`.

**Tila: laskentaa ja testattavuutta on parannettu, mutta järjestelmä ei ole
hyväksytty tuotantoon eikä kannattavuutta ole osoitettu.** Oracle-palvelimelle,
Alpacaan, Telegramiin tai oikean rahan kaupankäyntiin ei muodostettu yhteyttä.
`config.json`, tallennettu `models/ensemble.pkl` ja historialliset tulos-CSV:t
säilytettiin muuttamattomina. Uutta 100 osakkeen mallia ei koulutettu eikä otettu käyttöön.

## Mitä aiemmista korjauksista vahvistettiin?

Alkuperäiset 18 regressiotestiä läpäisivät jo ennen tämän työn muutoksia.
Centroidin asteikko opitaan koulutusaineistosta, vanha centroid vaatii
uudelleenkoulutuksen, puuttuva uutinen on nolla ja `backtest.py` käyttää
saman osakkeen seuraavaa saman New Yorkin päivän hintabaaria sekä erillistä
hintapolkua. Stopin läpi gappaaminen ja saman baarin stop/target-konflikti
on testattu. Paperin positionhaun virhe nostaa poikkeuksen.

Rajoitus: nämä ovat rajattujen funktioiden testejä, eivät todistus kaikkien
paperiprosessin haarojen turvallisuudesta. Aiempi Library-muistio käsitteli
main-haaran seitsemän osakkeen versiota; tämän haaran configissa on 100 osaketta.
Oracle-palvelimen käynnistetyn tiedoston vastaavuutta ei tarkastettu.

## Korjatut ongelmat

| Ongelma | Korjaus ja vaikutus |
|---|---|
| SPY-piirteet riippuivat osakkeen puuttuvista baareista | Benchmarkin tuotot, EMA ja ATR lasketaan sen omalla aikajanalla ennen liitosta. Puuttuva saman ajan SPY-rivi estää ennusteen. |
| Koulutuksen pitkä EMA-historia ja paperin 250 rivin historia erosivat | Täsmälleen viimeiseen 250 havaintoon rajattu EMA molemmille; 250 havaintoa vaaditaan. Piirreversio vaihtuu. |
| Noususarjan RSI oli NaN nollan keskitappion vuoksi | Noususarja saa RSI 100, tasainen sarja RSI 50. Tämä on määrittelykorjaus, ei tuottoon sovitettu raja. |
| Paperin uutispiirre oli eri kuin historian piirre | Yhteinen `news_features.py`: viimeksi saataville tullut tapahtuma, julkaisuajasta laskettu eksponentiaalinen vaimennus, neutraali puuttuva uutinen. Laskukaava ei ole enää normalisoitu 30 minuutin keskiarvo. |
| Julkaisu-, päivitys- ja vastaanottoaika sekoittuivat | Tapahtuma on käytettävissä vasta kaikkien kolmen ajan jälkeen. `received`-tila vaatii vastaanottoajan. Julkaisuaikaan perustuva historia on erikseen nimetty tutkimusproxy. |
| Uutispiirre ja erillinen kauppapisteytys saattoivat käyttää eri hetkeä | Pisteytys käyttää saman signaalirivin uutisarvoa kuin ML. Päätöshetki on baarin sulkeutuminen. |
| Keskeneräinen / vanha viimeinen baari saattoi kelvata | Eksplisiittinen `Sort.DESC`, yksi lisärivi keskeneräistä baaria varten, valmistuneiden baarien suodatus ja viimeisen odotetun baarin tarkistus. Ei paluuta vanhaan piirrekelpoiseen riviin. |
| Koulutus/testi jakautui eri aikaan eri osakkeilla | `train_models.py` vaatii yhteisen päivämäärärajan. `label_end` huomioi tuloksen todellisen valmistumisajan. Luokka ei ylitä päivää eikä puuttuvaa viiden minuutin väliä. |
| V6.5 poisti testirivejä tulevan targetin puuttuessa | Vain koulutus vaatii targetin. Ennuste ei riipu tulevien hintojen tai labelin saatavuudesta. |
| Vanha malli voitiin syöttää uusille piirteille | Mallipakettiin feature contract: järjestys, versio, historia, uutistila, uutisparametrit, benchmark ja feed. Integrated ja backtest hylkäävät yhteensopimattoman paketin. |
| Riskikoko ja 10 % / 50 % katot ohittivat toteutuskulut | Koko huomioi entryn ja stopin kustannukset, kulujen jälkeisen equityn ja käteisen. Gap-tappio voi silti ylittää stop-budjetin. |
| Puuttuva hinta saattoi synnyttää kuvitellun EOD-täytön | V6.5:n vakioajossa puuttuva baari avoimen position aikana keskeyttää koko ajon. Ei hiljaista hintojen täyttämistä. |
| Lyhennetty sessio käytti 15.50-sulkemista | Ajuri vaatii erillisen session kalenterin. Sulkemisproxy on 10 minuuttia ennen ilmoitettua 13.00/16.00 ET sulkua. Kalenterin oikeellisuus on tarkistettava lähteestä. |
| Signaalin mukana tullut kellosarake saattoi aikaistaa signaalia | Päivä ja kellonaika johdetaan alkuperäisestä aikaleimasta; saatavuuden on vastattava baarin loppua. |
| Target-gap saatettiin käsitellä myöhemmän intrabar-stopin jälkeen | Avaushetken target-gap tapahtuu ensin; jos kumpikaan ei täyty avauksessa, saman baarin stop voittaa. |
| Enum-muotoinen avoin toimeksianto saattoi jäädä tunnistamatta | Enum normalisoidaan. Tuntematon tai peruutusta odottava BUY käsitellään varauksena, kunnes tila on päättynyt. |
| Tyhjä salkku saatettiin julistaa päivän päätteeksi turvalliseksi avoimista BUY-toimeksiannoista huolimatta | Peruuta avoimet toimeksiannot ja varmista sekä positiot että toimeksiannot; myös sulkemispyynnön jälkeen. |

`commission_per_trade` tarkoittaa tässä edelleen yhtä edestakaisen kaupan
maksua, joka veloitetaan poistumisessa. Spread/slippage-bps veloitetaan
kummallekin puolelle. Simulaattorin `entry`/`exit` ovat kustannuksilla korjattuja
laskentahintoja, eivät väitteitä välittäjän toteutuneista limiittitäytöistä.

## Mitä oikeasti ajettiin?

| Tarkistus | Tulos | Mitä se EI osoita |
|---|---|---|
| Lähtöversion regressiot | 18/18 läpi | Oracle-prosessin oikeellisuutta |
| Laajennettu offline-testisarja, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 | 61/61 läpi | Historiallista tai tulevaa kannattavuutta |
| Sama testisarja, pandas 3.0.0 | 61/61 läpi | Alpacan SDK:n verkkointegraation toimivuutta |
| Uusi ajuri päästä päähän | Generoitu kahden osakkeen aineisto, oikea koulutus ja molemmat vertailuhaarat, raportit/tarkistussummat | 100 osakkeen markkinatulosta |
| Repon yhdeksän OHLCV-tiedoston rakenne | Kaikki läpäisivät numerot, aikaleimat, duplikaatit ja high/low-tarkistuksen | Oikeaa feediä, split-korjauksia, täydellisyyttä tai point-in-time-universumia |
| Seitsemän todellisen osaketiedoston piirrevastaavuus | Pitkän historian ja 250 rivin ikkunan viimeinen kelvollinen piirre täsmää 1e-12 toleranssilla | Koko aineiston tuottoa tai uutisennusteen laatua |
| Vanhalla ja uudella simulaattorilla ajettu synteettinen kokorajan vastatesti | Ennen 100 kpl / 10 005 USD; jälkeen 99 kpl / 9 904,95 USD | Strategian paranemista markkinadatalla |
| Aktiivisten moduulien syntaksi ja `git diff --check` | Läpi | Muiden vanhojen tutkimusohjelmien validointia |

Testit käynnistetään nimenomaisella listalla, ja verkkosoketit on estetty
ajon aikana. Kaupankäyntiprosessia ei tuoda moduulina: yksittäiset paperifunktiot
poimitaan AST:stä ja välittäjä korvataan mockilla. Mallitestit kouluttavat
pieniä muistissa olevia malleja; repon pickle-tiedostoa ei avata.

Tarkat testinimet, versiot, lähdetiedostojen SHA-256-tarkistussummat ja lokit:
`audit_evidence/pandas-2.2.3/` ja `audit_evidence/pandas-3.0.0/`.
Aineiston inventaario: `audit_evidence/data_inventory.json`.
Vastatesti: `audit_evidence/risk_counterexample.json`.

## Aineiston puute ja tulkinta

Tarkastetussa repossa / kehityshaarassa on **7/100 osakkeen hintatiedostot**, SPY
ja QQQ, sekä seitsemän uutis-CSV:tä. **93 osakkeen hintadata puuttuu** tästä
checkoutista. V6.4:n syys–lokakuun erillistä 100 osakkeen hakemistoa ei löytynyt.
GitHubin `v5.3-experimental`-haaran hakemistolista sisälsi saman seitsemän
osakkeen perusaineiston. Uutis-CSV:t eivät sisällä `received_at`-kenttää.
Käyttäjä kertoi laajemman datan ja vastaanottohistorialokin olevan olemassa,
mutta niiden repositorio-/tiedostopolkuja ei ollut käytettävissä.

Siksi koko V6.5:n historiallista vertailua, SPY-suodattimen tuottohyötyä tai
uutisten lisäarvoa **ei ole laskettu eikä vahvistettu**. Puuttuvia osakkeita ei
korvattu seitsemän osakkeen tuottovertailulla. Uusi ajuri pysähtyy puuttuvaan
aineistoon ennen koulutusta eikä anna osittaiselle universumille hyväksyttyä raporttia.

## Vielä ratkaisemattomat käyttöönoton esteet

1. Toimeksiannon epäselvä kuittaus, pysyvä `client_order_id`, ennen lähetystä
   tehtävä varaus, uudelleenkäynnistys ja välittäjään täsmäytys eivät vielä muodosta
   kattavasti testattua tilakonetta. Lähetyksen jälkeen tallentaminen jättää
   kaatumisikkunan. Näitä ei pidä pitää korjattuina enum-testien perusteella.
2. Opening-queue lähettää DAY-market-BUY:n, vaikka lokit puhuvat OPG:stä.
   Suojaava OCO luodaan myöhemmin. Osatäyttö, suojaamaton aikaväli, OCO-uusinta
   ja mahdollinen ylimääräinen sulkemistoimeksianto tarvitsevat erilliset
   tapahtumajärjestystestit. V6.5 ei simuloi tätä polkua.
3. Toteumaloki, osatäytöt, maksut ja koko tilin equity on vielä täsmäytettävä
   Alpacan toteumahistoriaan. Lähetysloki ei ole toteumaloki.
4. `walk_forward.py`, `nested_sensitivity_test.py` ja monet backup-/stress-skriptit
   käyttävät edelleen eri osakkeen seuraavan globaalin rivin entryä, eri
   allokaatioita tai puutteellista purgausta. Niitä ei ajettu eikä hyväksytty.
   Yhteinen uusi piirreversio ei tee niistä validoituja tutkimusmoottoreita.
   Vanhojen kolmen benchmark-sarakkeen kutsujen on siirryttävä raakaan OHLCV:hen.
5. Historiallisen 100 osakkeen valinta, listautumiset/poistumiset, splitit,
   adjusted/raw-käytäntö, IEX/SIP-ero ja kalenterin kattavuus on varmennettava.
6. Baarin open-täyttö ei mallinna havainto- tai laskentaviivettä, jonosijaa,
   markkinavaikutusta tai osatäyttöjä. Sulun MTM ei kata intrabar-drawdownia.
   Samanaikaisen stop/targetin konservatiivinen sääntö ei ratkaise kaikkia
   mahdollisia hintapolkuja.
7. Uutisfeedin katkos on eri asia kuin uutisten puuttuminen. Vastaanottolokin
   kattavuus ja lähteen terveystila pitää todentaa ennen uutistilan käyttöönottoa.
   Vanhoille tapahtumille ei saa keksiä vastaanottoaikoja.
8. Luokka on yhä +0,2 % kolmen viiden minuutin baarin jälkeen, ei 2R-voitto
   ennen stoppia. Kohteen muuttaminen olisi erillinen tutkimuskoe.

## Ajokomennot ja siirtyminen

```powershell
python tools/run_audit_tests.py --output-dir audit_evidence/local-check
python tools/audit_offline_evidence.py --output-dir audit_evidence/local-data-check
python v6_5_intraday_replay.py --help
python train_models.py --help
```

Koko vertailu, kun data ja tarkistettu kalenteri ovat saatavilla:

```powershell
python v6_5_intraday_replay.py --data-dir ../tradebot_v62_research/data --extension-dir ../tradebot_v62_research/data_v64 --sessions sessions-sep-oct-2026.csv --output-dir research_runs/v65-corrected-001
```

Kalenterissa sarakkeet `date,close`, esimerkiksi `2026-09-01,16:00` (ET).
Ajuri vaatii uuden tuloshakemiston. Se tallentaa molempien haarojen päivät,
kaupat, equityn, yhteenvedon, paritetun vertailun ja manifestin. Keskeytynyt
ajo jättää manifestin tilaan `started_not_validated`, ei valmiiksi tulokseksi.

Mallin koulutus vaatii erillisen uuden tiedostopolun. Vanha artifact ei ole
yhteensopiva uuden feature contractin kanssa. `train_models.py` kieltäytyy
kirjoittamasta `models/ensemble.pkl`:n päälle. Koulutus ei käynnistä paperibottia.
Älä kopioi kehitystiedostoja Oraclelle ennen erillistä hyväksyntää ja yllä olevien
käyttöönoton esteiden ratkaisemista.

Uutisten arviointi ja SPY-kokeen lukittu suunnitelma: `RESEARCH_PROTOCOL.md`.

## Ulkoiset määrittelylähteet

Tarkistettu 23.10.2026; koodin regressiot ovat yllä kuvattua omaa testinäyttöä.

- Alpaca: baarin aikaleima on aikavälin alku: https://docs.alpaca.markets/us/docs/market-data-faq
- Alpaca: oletusjärjestys, limit ja sivutus: https://docs.alpaca.markets/us/reference/stockbars
- Alpaca SDK `Sort.DESC`: https://alpaca.markets/sdks/python/api_reference/common/enums.html
- Alpaca SDK toimeksiantojen peruutus: https://alpaca.markets/sdks/python/api_reference/trading/orders.html
- Paperitäyttöjen rajoitukset: https://docs.alpaca.markets/us/v1.4.2/docs/paper-trading
