# Toimeksiantojen kovennus 26.10.2026

Haara: `audit-hardening-v1`. Lähtöcommit: `ac0741d`.
Tämä jatkaa [23.10. auditointia](AUDIT_REVIEW_2026-10-23.md).
**Kehitysversio ei ole vielä tuotantovalmis. Oracle-palvelimeen, käynnissä
olevaan paperbottiin tai oikean rahan kaupankäyntiin ei tehty muutoksia.**

## Korjatut ongelmat

Aiempi ostotoimeksianto lähetettiin ennen pysyvää tallennusta. Vastauksen
kadotessa tai prosessin kaatuessa sama ostos voitiin yrittää uudelleen.
Avauskaupan suojaus riippui suljettujen toimeksiantojen listasta ja
ilmoituspolusta. Keskeneräiset osatäytöt eivät olleet kattavasti mukana.
OCO-vastauksen epäselvyys saattoi johtaa uusiin suojausyrityksiin ja lopulta
markkinamyyntiin ilman varmuutta aiemman myynnin tilasta.

Muutokset:

- `order_journal.py`: tilikohtainen SQLite-loki, deterministinen
  `client_order_id` (tili/päivä/osake/rooli), transaktio ja `synchronous=FULL`.
  Lähetysaikomus tallennetaan **ennen** välittäjäkutsua. Varattu tunnus ei
  oikeuta toiseen lähetykseen edes aikakatkaisun, 404:n tai restartin jälkeen.
  Epäselvä aikomus estää uudet ostot kaikissa osakkeissa.
- `order_lifecycle.py`: suora tunnushaku, kumulatiivisten täyttöjen ja
  välittäjän identiteetin tarkistus. Tuntematon tila, määrän pieneneminen,
  vaihtunut tunnus tai ristiriitainen positio estää uudet ostot.
  `replaced` ei ole tuettu lopputila: seuraajatoimeksianto pitää selvittää.
- Osittain täyttyneestä ostosta pyydetään jäljellä olevan määrän peruutus
  kerran. Peruutuskuittaus ei tarkoita lopullista peruutusta. Avauskaupan OCO
  luodaan vasta lopputilan ja positiomäärän täsmättyä. Myöhäinen osatäyttö
  sisältyy suojattavaan määrään ja painotettuun toteutushintaan.
- OCO-suojauksen tuntematon kuittaus ei käynnistä uutta OCO:ta eikä
  ilmoituspolun hätämyyntiä. Näkyvän stopin hinnan ja jäljellä olevan määrän
  pitää vastata tallennettua pyyntöä ja positiota. Liian suuri aktiivinen
  myyntimäärä, short-positio tai ylimyynti estää uudet ostot.
- Suojauksen ylläpito toimii erillään Telegram-ilmoituksista, myös
  päiväkiintiön täytyttyä. Taustatarkistus tehdään enintään kerran viidessä
  sekunnissa onnistuneen kierroksen jälkeen; jokainen uusi osto pakottaa
  tuoreen tarkistuksen. Tunnettu välittäjän tunnus haetaan suoraan, mikä
  vähentää API-kuormaa. Tämä ei ole viiden sekunnin vasteaikatakuu.
- `paper_order_adapter.py`: SDK:n sisäiset HTTP-uudelleenyritykset poistetaan,
  yhteys-/lukuaikarajat ovat 5/10 sekuntia. `alpaca-py==0.44.0` lukitaan,
  koska muutos käyttää version tarkistettuja sisäisiä transport-koukkuja.
  Asiakas muodostetaan aina paper-osoitteeseen. Täyden 500 toimeksiannon
  listan tapauksessa mahdollinen sivutuskatko estää toiminnan.
- Päiväkiintiö ja osakekohtainen yhden yrityksen raja palautuvat lokista.
  Myös hylätty tai epäselvä lähetysyritys kuluttaa kiintiön konservatiivisesti.
  Lokin varaus täydentää välittäjän positio-/avointen ostojen näkyvyyttä:
  osakekohtainen enimmäinen varaus estää saman position kahteen kertaan
  laskemisen. Täyden alkuperäisen ostomäärän varaus säilyy päivän ajan,
  paitsi varmasti lopullisesti nollatäyttöisen toimeksiannon kohdalla.
- Päivän riskitila ja avausjono kirjoitetaan atomisesti; vioittunut tila
  ei nollaudu hiljaisesti. Avausjonoa ei poisteta päivän vaihtuessa.
  Vanhan päivän jonoehdokkaita ei lähetetä. Uusi loki vaatii tyhjän erillisen
  paper-tilin; vanhoja lähetettyjä jonotietoja ei oteta automaattisesti haltuun.
- Prosessilukko estää kaksi kirjoittajaa samassa työhakemistossa.
  Suljettu kauppa poistuu aktiivisesta täsmäytyksestä vasta, kun myyntien
  kokonaismäärä, lopulliset sisartoimeksiannot ja tyhjä positio täsmäävät.
- Uuden oston portti tarkistaa välittäjän aukiolon ja EOD-ikkunan uudelleen.
  Sadan osakkeen pitkä skannaus ei saa ohittaa päivän sulkurajaa.

Strategian ML-, SPY-, stop-, tavoite-, riski- tai osakevalintaparametreja
**ei optimoitu eikä muutettu**. Osatäytön loppuosan peruutus ja konservatiiviset
varaukset muuttavat toteutuskäyttäytymistä turvallisuussyistä; niiden ei väitetä
parantavan tuottoa. Avauskauppa on edelleen DAY-markkinaosto aukeamisen jälkeen,
ei aito OPG-huutokauppatoimeksianto.

## Todella testattu

`test_order_lifecycle.py` lisää 35 testiä aiempaan 61 testiin.
Kaikki 96 testiä läpäisivät molemmat paikalliset ajot:

| Python | NumPy | pandas | alpaca-py | Testit | Virheet/hylätyt |
|---|---|---|---|---|---|
| 3.12.14 | 2.3.5 | 2.2.3 | 0.44.0 | 96 | 0 / 0 |
| 3.12.14 | 2.3.5 | 3.0.0 | 0.44.0 | 96 | 0 / 0 |

Ajot: `python tools/run_audit_tests.py --output-dir <hakemisto>`.
Ajuri estää ulospäin suuntautuvat socket-yhteydet. Paper-prosessia ei tuoda
moduulina, vanhaa mallipickleä ei ladata eikä välittäjään muodosteta yhteyttä.
Oikeat SDK-pyyntöoliot ja HTTP-transport testataan korvatuilla vastauksilla.
Prosessin kriittiset funktiot testataan AST:stä irrotettuina riippuvuuksilla,
sekä kaikki muuttuneet Python-moduulit tarkistetaan syntaksin osalta.

Testatut virhetilanteet sisältävät kadonneen kuittauksen ennen/jälkeen
hyväksynnän, kaatumisen varauksen ja lähetyksen välissä, levyvirheet,
tiliristiriidan, rinnakkaiset varaukset, prosessilukon, restartin,
404:n, kumulatiivisen täyttömäärän regression, tuntemattoman tilan,
myöhäiset osatäytöt, peruutuksen aikakatkaisun, position näkyvyysviiveen,
olemassa olevan myynnin, OCO-kuittauksen katoamisen, väärän stopin,
osittaisen take profitin, ylimyynnin, vanhan avausjonon ja päiväkiintiön
palautumisen. Myös prosessin oikeat intraday- ja avauslähetysfunktiot
ajetaan valevälittäjällä: kadonnut kuittaus ei aiheuta toista lähetystä.

Pysyvät ajotulokset ja testattujen Python-lähteiden SHA-256:t:

- [pandas 2.2.3 JSON](audit_evidence/2026-10-26/pandas-2.2.3/test_results.json)
  ja [lokit](audit_evidence/2026-10-26/pandas-2.2.3/test_results.txt).
- [pandas 3.0.0 JSON](audit_evidence/2026-10-26/pandas-3.0.0/test_results.json)
  ja [lokit](audit_evidence/2026-10-26/pandas-3.0.0/test_results.txt).

Kehityksen ensimmäinen kokonaisajo epäonnistui kahteen tarkistukseen:
testissä SDK:n enum-osoite luettiin merkkijonona arvon sijaan, ja
ostorajan lisäys muutti palautettua listaa paikallisen kopion sijaan.
Molemmat korjattiin; yllä olevat tulokset ovat lopullisista lähteistä.
GitHub Actionsin sama matriisi ajetaan kehityshaaran pushista; tämän dokumentin
paikalliset tulokset eivät yksin ole väite CI:n onnistumisesta.

## Ratkaisematta / käyttöönoton esteet

1. **EOD:n `close_all_positions(cancel_orders=True)` on edelleen vanha
   massatoiminto ilman pysyvää sulkuaikomusta ja kuittauksen täsmäytystä.**
   Aikakatkaisun jälkeistä uudelleenyritystä ei ole osoitettu turvalliseksi.
   EOD-sulun täyttöjä ei vielä liitetä uuteen kauppakohtaiseen lokiin:
   tämän vuoksi seuraavan päivän ostot voivat estyä turvallisesti
   position ja kirjattujen poistumisten ristiriitaan. Tätä ei saa kiertää
   poistamalla lokia. Seuraava korjaus on yksilöity, pysyvästi kirjattu
   EOD-sulku sekä seuraavan päivän palautumisen testaus.
2. Peruutetun osittaisen **intraday-bracketin** lapsitoimeksiantojen
   aktivoitumista ei ole vahvistettu välittäjällä. Koodi estää uudet ostot,
   jos toimivaa stoppia ei voida todentaa; kilpailevaa OCO:ta ei lisätä.
   Myös lopullisesti hylätty OCO vaatii selvityksen eikä automaattista
   uudelleenyritystä. Osatäytön ja suojauksen välissä on suojaamaton aikaväli.
3. Simuloidut välittäjävastaukset eivät todista Alpacan todellista
   tapahtumajärjestystä, OCO:n kilpailutilanteita, täyttölaatua, verkon
   saatavuutta tai yhteistä käyttöä muiden prosessien kanssa. Prosessilukko
   koskee yhtä hakemistoa, ei eri koneita. Tallennuksen laite-/virtakatkoja
   ei ole fyysisesti testattu. Loki edellyttää pysyvää paikallista levyä.
4. Täysi paper-prosessin käynnistys, viikon mittainen valvottu paper-ajo ja
   todelliset osatäytöt ovat testaamatta. Aiemman mallisopimuksen
   käyttöönottoesteet pysyvät voimassa. SDK-päivitys vaatii transport-testit.
5. Tässä muutoksessa ei ajettu uutta tuottovertailua, uutisetuvertailua tai
   SPY +0,10 % -kokeilua. [Tutkimusprotokolla](RESEARCH_PROTOCOL.md) ja
   aiemmin todetut 100 osakkeen historiadatan / vastaanottoaikojen puutteet
   eivät ratkea toimeksiantotesteillä. Kannattavuuden paranemista ei ole osoitettu.

Oracleen ei kirjauduttu, palvelua ei käynnistetty/uudelleenkäynnistetty,
avaimia ei luettu eikä tilille lähetetty toimeksiantoja. Muutoksia ei yhdistetä
pää-/tuotantohaaraan tämän työn perusteella.

## Tarkistetut rajapintalähteet

- Alpaca toimeksiannot ja bracket/OCO-semanttiikka:
  https://docs.alpaca.markets/us/docs/orders-at-alpaca
- SDK-pyyntömallit:
  https://alpaca.markets/sdks/python/api_reference/trading/requests.html
- SDK-toimeksiantorajapinta:
  https://alpaca.markets/sdks/python/api_reference/trading/orders.html
- Lisäksi paikallisesti asennetun `alpaca-py==0.44.0`:n `RESTClient._request`,
  `RESTClient._one_request` ja `TradingClient.get_order_by_client_id`
  tarkistettiin lähdekoodista. Pelkkä dokumentaatio ei korvaa yllä eroteltua
  välittäjän integraatiotestausta.
