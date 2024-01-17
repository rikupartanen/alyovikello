'use strict';

//Komentojen nimet alkavat friendlyeatseina koska jostakin syystä sen muuttaminen rikkoi koodin :D
function FriendlyEats() {
  this.filters = {};
  var that = this;

  firebase.firestore().enablePersistence()
    .then(function() {
      return firebase.auth().signInAnonymously();
    })
    .then(function() {
      that.initTemplates();
      that.initRouter();
      that.initReviewDialog();
      that.initFilterDialog();
    }).catch(function(err) {
      console.log(err);
    });
}

//Käynnistetään routing
FriendlyEats.prototype.initRouter = function() {
  this.router = new Navigo();
  var that = this;
  this.router
    .on({
      '/': function() {
        that.updateQuery(that.filters);
      }
    })
    .on({
      '/restaurants/*': function() {
        var path = that.getCleanPath(document.location.pathname);
        var id = path.split('/')[2];
        that.viewRestaurant(id);
      }
    })
    .resolve();

  firebase
    .firestore()
    .collection('restaurants')
    .limit(1)
    .onSnapshot(function(snapshot) {
      if (snapshot.empty) {
        that.router.navigate('/setup');
      }
    });
};

// Sivun käynnistys
window.onload = function() {
  window.app = new FriendlyEats();
};
