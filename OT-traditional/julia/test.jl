function square(x)
    return x^2
end

function greet(name)
    println("Hello, $name !")
end

x = 5
println("The square of $x is ", square(x))

name = "Julia"
greet(name)
