FROM node:20-alpine AS build
WORKDIR /frontend
COPY platform/frontend/package.json /frontend/package.json
RUN npm install
COPY platform/frontend /frontend
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /frontend/dist /usr/share/nginx/html
COPY platform/frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
