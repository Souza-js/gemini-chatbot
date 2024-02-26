const path = require('path');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
  entry: './index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js' // Chunk hashing
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      },
      {
        test: /\.(png|svg|jpg|gif)$/,
        use: [
          'file-loader',
          {
            loader: 'image-webpack-loader', // Optimize images
            options: {
              mozjpeg: {
                progressive: true,
                quality: 65
              },
              optipng: {
                enabled: true,
              },
              pngquant: {
                quality: [0.65, 0.90],
                speed: 4
              },
              gifsicle: {
                interlaced: true,
              },
              webp: {
                quality: 75
              }
            }
          },
        ],
      },
    ]
  },
  mode: 'production', // Enable production mode
  optimization: {
    minimizer: [
      new TerserPlugin(), // Minify JavaScript
    ],
  },
};
