const path = require('path');

module.exports = function (app) {
  app.use((req, res, next) => {
    const ext = path.extname(req.url);
    if (ext === "" || ext === ".html" || ext === ".js" || ext === ".wasm") {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    }
    next();
  });
}
