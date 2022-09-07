var items = document.querySelectorAll("img.NO-CACHE");
for (var i = items.length; i--;) {
    var img = items[i];
    img.src = img.src + '?' + Date.now();
}