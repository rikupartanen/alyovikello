'use strict';

//Haetaan kuvat ja niiden tiedot, unohdin muuttaa koodin laittamaan ne ajan perusteella järjestykseen
FriendlyEats.prototype.getAllRestaurants = function (render) {
  const query = firebase.firestore()
    .collection('restaurants')
    .orderBy('avgRating', 'desc')
    // Vaihda tuo ajaksi
    .limit(50);
  this.getDocumentsInQuery(query, render);
};

//Renderöidään kuvat ja niiden tiedot oikein
FriendlyEats.prototype.getDocumentsInQuery = function (query, render) {
  query.onSnapshot((snapshot) => {
    if (!snapshot.size) {
      return render();
    }

    //Uuden kuvan tullessa näytetään sekin
    snapshot.docChanges().forEach((change) => {
      if (change.type === 'added' || change.type === 'modified') {
        render(change.doc);
      }
    });
  });
};
