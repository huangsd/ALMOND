%parameters
function result = para(para1, para2, para3, para4, para5)

l1 = length(para1);
l2 = length(para2);
l3 = length(para3);
l4 = length(para4);
l5 = length(para5);
x=1;
for i=1:l1
    for j=1:l2
        for k=1:l3
            for l = 1:l4
                for p = 1:l5
                    result(x,:)=[para1(i), para2(j), para3(k), para4(l), para5(p)];
                    x=x+1;      
                end
                    
            end
        end
    end
end

end