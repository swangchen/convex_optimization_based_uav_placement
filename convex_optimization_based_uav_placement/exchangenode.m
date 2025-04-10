function G = exchangenode(G,a,b)


buffer=G(:,a);
G(:,a)=G(:,b);
G(:,b)=buffer;


buffer=G(a,:);
G(a,:)=G(b,:);
G(b,:)=buffer;