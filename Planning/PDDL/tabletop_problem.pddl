(define (problem coffee_task)
  (:domain tabletop)
  (:objects
    coffee_pod drawer coffee_dispenser mug table
  )
  (:init
    (inside coffee_pod drawer)
    (on mug table)
  )
  (:goal
    (and
      (installed coffee_pod coffee_dispenser)
      (under mug coffee_dispenser)
    )
  )
)
